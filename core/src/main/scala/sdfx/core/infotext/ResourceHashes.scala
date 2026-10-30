package sdfx.core.infotext

import scala.util.control.NonFatal

/** Civitai-compatible resource hash metadata assembled from explicit facts.
  * This formatter never guesses resource identity or reads a model file.
  */
object ResourceHashes:
  def merge(existing: Option[String], facts: Seq[(String, String)]): String =
    val prior = existing.filter(_.nonEmpty).map { text =>
      try
        ujson.read(text) match
          case obj: ujson.Obj => obj.value.toSeq.map {
            case (key, ujson.Str(value)) => key -> value
            case _ => throw new IllegalArgumentException("Hashes values must be strings")
          }
          case _ => throw new IllegalArgumentException("Hashes must be a JSON object")
      catch
        case error: IllegalArgumentException => throw error
        case NonFatal(error) => throw new IllegalArgumentException("Invalid Hashes JSON", error)
    }.getOrElse(Nil)
    def validated(entry: (String, String)): (String, String) =
      val (key, hash) = entry
      require(key.nonEmpty && !key.exists(_.isControl), "Resource key must be nonempty and contain no control characters")
      require(hash.matches("[0-9a-fA-F]{8,64}"), s"Invalid resource hash for $key")
      key -> hash.toLowerCase(java.util.Locale.ROOT).take(10)
    val previous = prior.map(validated).toMap
    val current = facts.map(validated)
    current.groupBy(_._1).foreach { (key, entries) =>
      require(entries.map(_._2).distinct.size == 1, s"Conflicting resource hashes for $key")
    }
    ujson.Obj.from((previous ++ current).toSeq.sortBy(_._1).map((key, value) => key -> ujson.Str(value))).render()
