package sdfx.engine

import sdfx.core.model.GenerationParams
import sdfx.core.payload.*
import sdfx.core.plugins.api.*
import sdfx.core.protocol.{GenerationResult, Json, LoadedModel}

/** Native submission metadata is separate from the exact protocol body. */
private[engine] object PayloadCaptureContext:
  def plugin(plugin: Plugin, values: Values): ujson.Obj = ujson.Obj(
    "id" -> plugin.id, "title" -> plugin.name,
    "values" -> ujson.Obj.from(values.map.toSeq.sortBy(_._1).map((k, v) => k -> ujson.Str(v))))

  def submitted(job: Job): SubmittedPayloadContext = job.submittedContext match
    case Some(context) => context.copy(origin = job.origin, sourceMode = context.sourceMode.filter(RequestPayloadStore.KnownModes))
    case None =>
      val configuration = ujson.Obj("ui_forms" -> "not_available_for_origin",
        "has_mask" -> job.params.img2img.flatMap(_.mask).exists(_.nonEmpty),
        "selected_script" -> job.script.map((p, v) => plugin(p, v)).getOrElse(ujson.Null),
        "always_on" -> ujson.Arr.from(job.alwaysOn.map { (p, v) =>
          val result = plugin(p, v)
          result("active") = p.isActive(v)
          result
        }))
      val json = PayloadJsonBudget.write(configuration).getOrElse("{\"status\":\"unavailable_limits\"}")
      SubmittedPayloadContext(job.origin,
        Option.when(job.origin == "mcp")(if job.params.isImg2Img then "img2img" else "txt2img"), Some(json))

  def document(submitted: SubmittedPayloadContext, limits: PayloadJsonBudget.Limits): Either[String, ujson.Obj] =
    val configuration = submitted.configurationJson match
      case None => Right(ujson.Null)
      case Some(json) => PayloadJsonBudget.admit(json, limits)(ujson.read(json))
    configuration.map { value =>
      ujson.Obj("body_schema" -> "sdfx protocol v1 generation params; native null/default fields are retained",
        "submitted" -> ujson.Obj("origin" -> submitted.origin,
          "source_mode" -> submitted.sourceMode.map(ujson.Str(_)).getOrElse(ujson.Null),
          "ui_variation_enabled" -> submitted.uiVariationEnabled.map(ujson.Bool(_)).getOrElse(ujson.Null),
          "configuration" -> value))
    }

  def captured(submitted: SubmittedPayloadContext, p: GenerationParams, at: Long, model: Option[LoadedModel],
      limits: PayloadJsonBudget.Limits): Either[String, String] =
    document(submitted, limits).flatMap { context =>
      context("reported_at_capture") = ujson.Obj("captured_at_millis" -> at,
        "meaning" -> "currently reported model snapshot, not proof of the model used by concurrent execution",
        "model" -> model.map(Json.toJs(_)).getOrElse(ujson.Null))
      context("requested_seed_options") = ujson.Obj("seed" -> Json.toJs(p.seed), "subseed" -> Json.toJs(p.subseed),
        "subseed_strength" -> p.subseedStrength, "extras_requested" -> (p.subseed != -1 || p.subseedStrength != 0),
        "extras_requested_basis" -> "native_params_inference",
        "seed_resize" -> "not_in_native_schema")
      PayloadJsonBudget.write(context, limits)
    }

  def completion(result: GenerationResult, limits: PayloadJsonBudget.Limits): Either[String, Option[String]] =
    val fields = Seq("model_name" -> result.modelName, "model_hash" -> result.modelHash,
      "vae_name" -> result.vaeName, "vae_hash" -> result.vaeHash).flatMap((key, value) => value.map(v => key -> ujson.Str(v)))
    if fields.isEmpty then Right(None)
    else PayloadJsonBudget.write(ujson.Obj.from(fields), limits).map(Some(_))
