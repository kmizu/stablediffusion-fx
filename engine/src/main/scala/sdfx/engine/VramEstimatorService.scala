package sdfx.engine

import sdfx.core.config.AtomicFiles
import sdfx.core.protocol.{EngineEvent, InterruptionDecision, InterruptionProtection, Protocol}
import java.nio.file.{Files, Path}
import scala.concurrent.{ExecutionContext, Future}

/** Each calibration owns its interrupt target independently of normal generation jobs. */
private[engine] final class VramEstimatorSession:
  private var active: Option[String] = None
  def current: Option[String] = synchronized(active)
  def start(): Option[String] = synchronized {
    if active.nonEmpty then None
    else
      active = Some(java.util.UUID.randomUUID().toString)
      active
  }
  def finish(owner: String): Unit = synchronized {
    if active.contains(owner) then active = None
  }

object VramEstimatorHistory:
  /** An interrupted run with no completed case cannot supersede usable measurements. */
  def newestMeasured(records: Seq[ujson.Value], fingerprint: String): Option[ujson.Value] =
    records.reverseIterator.find(record => record("fingerprint").str == fingerprint &&
      record("rows").arr.exists(_("status").str == "complete"))
  val MaxBytes = 16 * 1024 * 1024
  def validate(record: ujson.Value): Unit =
    val obj = record.obj
    require(obj.get("version").contains(ujson.Num(1)), "Unsupported VRAM profile version")
    require(obj.get("status").exists(v => Set("running", "complete", "interrupted", "stopped_after_error").contains(v.str)), "Invalid VRAM profile status")
    require(record("fingerprint").str.matches("[0-9a-f]{24}"), "Invalid VRAM profile fingerprint")
    require(record("identity").obj != null, "Missing VRAM profile identity")
    require(record("profile_path").str.nonEmpty, "Missing VRAM profile identifier")
    val rows = record("rows").arr
    require(rows.size <= 4096, "Too many VRAM profile rows")
    rows.foreach { row =>
      require(Set("txt2img", "txt2img_hr", "txt2img_hr_latent", "img2img").contains(row("mode").str), "Invalid VRAM profile mode")
      require(Set("complete", "oom", "error").contains(row("status").str), "Invalid VRAM profile row status")
      if row("status").str == "complete" then
        Seq("pixels", "batch_size", "reserved_delta_mb").foreach { field =>
          val value = row(field).num
          require(value.isFinite && value >= 0 && (field == "reserved_delta_mb" || (value > 0 && value == math.floor(value))), s"Invalid profile $field")
        }
    }
  def merge(existing: Seq[ujson.Value], record: ujson.Value, maxRecords: Int = 32, maxBytes: Int = MaxBytes): Seq[ujson.Value] =
    require(maxRecords > 0 && maxBytes > 0)
    validate(record)
    val key = record("profile_path").str
    var result = (existing.filterNot(_("profile_path").str == key) :+ ujson.read(record.render())).takeRight(maxRecords)
    def size: Int = ujson.Arr.from(result).render().getBytes(java.nio.charset.StandardCharsets.UTF_8).length
    while result.size > 1 && size > maxBytes do result = result.tail
    require(size <= maxBytes, "VRAM profile exceeds history size limit")
    result
  def load(path: Path): Seq[ujson.Value] =
    if !Files.exists(path) then Nil
    else
      require(Files.size(path) <= MaxBytes, "VRAM history exceeds 16 MiB")
      val records = ujson.read(Files.readString(path)).arr.toSeq
      require(records.size <= 32, "Too many VRAM history records")
      records.foreach(validate)
      records

final class VramEstimatorService(services: AppServices):
  private given ExecutionContext = ExecutionContext.global
  private val session = new VramEstimatorSession
  private var activeTransport: Option[(String, ExtrasBatchHandle)] = None
  val historyFile: Path = services.home.configDir.resolve("vram-estimator-profiles.json")
  def isBenchmarking: Boolean = session.current.nonEmpty
  def history(): Seq[ujson.Value] = VramEstimatorHistory.load(historyFile)
  def remember(record: ujson.Value): Seq[ujson.Value] = synchronized {
    val records = VramEstimatorHistory.merge(history(), record)
    AtomicFiles.writeString(historyFile, ujson.Arr.from(records).render())
    records
  }
  def profile(): Future[ujson.Value] = services.client.raw(Protocol.Types.VramEstimator, ujson.Obj("operation" -> "profile"))
  def exportProfile(record: ujson.Value, path: Path): Unit =
    VramEstimatorHistory.validate(record)
    val encoded = record.render()
    require(encoded.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= VramEstimatorHistory.MaxBytes, "Profile exceeds 16 MiB")
    AtomicFiles.writeString(path, encoded)
  def importProfile(path: Path): Future[Seq[ujson.Value]] =
    Future {
      require(Files.size(path) <= VramEstimatorHistory.MaxBytes, "Profile exceeds 16 MiB")
      val record = ujson.read(Files.readString(path))
      VramEstimatorHistory.validate(record)
      record
    }.flatMap { record =>
      services.client.raw(Protocol.Types.VramEstimator, ujson.Obj("operation" -> "profile", "profile" -> record))
        .flatMap(_ => Future(remember(record)))
    }
  def estimate(record: ujson.Value, mode: String, width: Int, height: Int, batch: Int): Future[ujson.Value] =
    VramEstimatorHistory.validate(record)
    services.client.raw(Protocol.Types.VramEstimator, ujson.Obj("operation" -> "estimate", "profile" -> record,
      "mode" -> mode, "width" -> width, "height" -> height, "batch" -> batch))
  def interrupt(): Future[InterruptionDecision] = synchronized(activeTransport.map(_._2)) match
    case Some(transport) => transport.cancel()
    case None => Future.successful(InterruptionDecision(true, false, InterruptionProtection()))
  def benchmark(config: ujson.Obj, listener: EngineEvent => Unit = _ => ()): Future[ujson.Value] =
    val admitted = synchronized {
      session.start().map { owner =>
        val control = new ExtrasBatchControl(
          () => services.supervisor.interruptAfterStartupOwned(owner, restartOnTimeout = false),
          () => services.client.interruptionProtection())
        val transport = services.client.extrasBatchOwned(owner, control, listener, restartOnTimeout = false)
        activeTransport = Some(owner -> transport)
        owner -> transport
      }
    }
    admitted match
      case None => Future.failed(new IllegalStateException("A VRAM calibration is already active"))
      case Some((owner, transport)) =>
        def release(): Unit =
          transport.finish()
          synchronized {
            activeTransport.filter(_._1 == owner).foreach(_ => activeTransport = None)
            session.finish(owner)
          }
        if !services.queue.isIdle then
          release()
          Future.failed(new IllegalStateException("Wait for generation to finish before VRAM calibration"))
        else
          val operation = Future {
            history() // Reject corrupt history before starting an expensive calibration.
            val params = ujson.read(config.render()).obj
            params("operation") = ujson.Str("benchmark")
            params("session_id") = ujson.Str(owner)
            params("output_dir") = ujson.Str(services.home.outputs(services.settings).toString)
            ujson.Obj.from(params)
          }.flatMap { params =>
            transport.request(Protocol.Types.VramEstimator, params)
          }.flatMap { record => Future { remember(record); record } }.recoverWith { case original =>
            profile().flatMap { response =>
              val partial = response.obj.get("profile").filter(_ != ujson.Null).filter { record =>
                record.obj.get("config").exists(_.obj.get("session_id").contains(ujson.Str(owner)))
              }
              partial match
                case Some(record) => Future(remember(record)).transformWith(_ => Future.failed(original))
                case None => Future.failed(original)
            }.recoverWith { case _ => Future.failed(original) }
          }
          operation.andThen { case _ => release() }(ExecutionContext.parasitic)
