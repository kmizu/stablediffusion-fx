package sdfx.mcp

import sdfx.core.plugins.api.*
import scala.jdk.CollectionConverters.*

/** Per-request values use declared defaults, never hidden desktop image paths. */
private[mcp] final class AlwaysOnInputs(
    plugins: () => Seq[Plugin], choices: ChoiceSource => Seq[String],
    image: (String, String) => String, localFile: (String, String) => String,
    directory: Option[(String, String) => String] = None,
    activeControlNet: Boolean = false
):
  def available(tab: Tab): Seq[Plugin] = plugins().filter(p => p.kind == PluginKind.AlwaysOn && p.tabs.contains(tab))

  def schema(plugin: Plugin): ujson.Obj =
    Schema.obj(plugin.controls.filter(_.hasValue).map { c =>
      val value: ujson.Value = c match
        case Control.Slider(_, label, min, max, _, d) => Schema.num(label, Some(min), Some(max), Some(d))
        case Control.IntField(_, label, min, max, d) => Schema.int(label, Some(min), Some(max), Some(d))
        case Control.Check(_, label, d) => Schema.bool(label, Some(d))
        case v: Control.Choice => Schema.str(v.label, if v.allowCustom then Nil else v.options, Some(v.default))
        case Control.DynamicChoice(_, label, src, d, allowNone) =>
          Schema.str(label, (if allowNone then Seq("", "None") else Nil) ++ choices(src), Some(d))
        case Control.ImageFile(_, label, multiple) => Schema.str(s"$label: allowed local path or base64/data URI${if multiple then "; one per line" else ""}")
        case _ => Schema.str(c.label, default = Some(c.default))
      c.key -> value
    })

  def parse(obj: ujson.Obj, tab: Tab): Seq[(Plugin, Values)] =
    if obj.value.size > 128 then throw ToolError("always_on: at most 128 plugins per request")
    val requested = obj.value.toMap
    val known = available(tab)
    requested.keys.foreach { id =>
      if !known.exists(_.id == id) then throw ToolError(s"always_on: unknown, disabled, or unsupported plugin '$id'; use list_always_on")
    }
    // Registry order is the same order as the GUI, independent of JSON key order.
    known.filter(p => requested.contains(p.id)).map { plugin =>
      val values = requested(plugin.id) match
        case o: ujson.Obj => o
        case _ => throw ToolError(s"always_on.${plugin.id} must be an object of control values")
      plugin -> validate(plugin, values)
    }

  def validate(plugin: Plugin, values: ujson.Obj, prefix: String = "always_on"): Values =
    val controls = plugin.controls.filter(_.hasValue)
    val args = Args(values, controls.map(_.key))
    val checked = controls.map { control =>
      val key = control.key
      val field = s"$prefix.${plugin.id}.$key"
      val value = control match
        case Control.Slider(_, _, min, max, _, d) => Values.fmt(args.num(key, min, max).getOrElse(d))
        case Control.IntField(_, _, min, max, d) => args.int(key, min, max).getOrElse(d).toString
        case Control.Check(_, _, d) => args.bool(key).getOrElse(d).toString
        case c: Control.Choice =>
          val v = args.str(key).getOrElse(c.default)
          if !c.allowCustom && !c.options.contains(v) then throw ToolError(s"$field: invalid choice; use list_always_on")
          v
        case Control.DynamicChoice(_, _, src, d, allowNone) =>
          val v = args.str(key).getOrElse(d)
          if args.has(key) && !(allowNone && (v.isEmpty || v == "None")) && !choices(src).contains(v) then
            throw ToolError(s"$field: invalid model/choice; use list_always_on")
          v
        case Control.ImageFile(_, _, multiple) =>
          val v = args.str(key).getOrElse("").trim
          if v.isEmpty then ""
          else if multiple then v.split("[\\r\\n]+").filter(_.nonEmpty).map(image(_, field)).mkString("\n")
          else image(v, field)
        case _ =>
          val v = args.str(key).getOrElse(control.default)
          if plugin.id == "inpaint_difference" && Set("painted_path", "altered_path")(key) && v.nonEmpty then image(v, field)
          else if plugin.id == "controlnet_fastload" && key == "path" && v.nonEmpty then
            def setting(name: String): String = args.str(name).getOrElse(controls.find(_.key == name).map(_.default).getOrElse(""))
            if setting("mode") == "save_only" || (activeControlNet && setting("priority") == "current_first") then ""
            else localFile(v, field)
          else if key.endsWith("_custom_model") || key == "head" || key == "image_encoder" then
            val repo = v.stripPrefix("hf://")
            val remoteEncoder = key == "image_encoder" && repo.matches("[A-Za-z0-9_-][A-Za-z0-9_.-]*/[A-Za-z0-9_-][A-Za-z0-9_.-]*")
            if v.isEmpty then ""
            else if remoteEncoder then "hf://" + repo
            else localFile(v, field)
          else if Set("frames_dir", "output_dir")(key) then
            val p = java.nio.file.Paths.get(v)
            if p.isAbsolute || PathPolicy.isNetworkPath(v) || p.iterator().asScala.exists(_.toString == "..") then
              throw ToolError(s"$field must be a relative output subdirectory without '..'")
            v
          else if key == "wildcard_dirs" && v.trim.nonEmpty then
            v.split("[;\\r\\n]+").map(_.trim).filter(_.nonEmpty).map(directory.getOrElse(localFile)(_, field)).mkString(";")
          else v
      key -> value
    }.toMap
    Values(checked)
