package sdfx.mcp

import scala.util.Try

/** Explicit, bounded interpretations of metadata emitted by native plugins.
  * These do not execute plugins and do not claim to be desktop form restoration.
  */
private[mcp] object PluginMetadataCandidates:
  private def number(fields: Map[String, String], key: String, min: Double, max: Double): Option[Double] =
    fields.get(key).flatMap(_.toDoubleOption).filter(v => v.isFinite && v >= min && v <= max)
  private def integer(fields: Map[String, String], key: String, min: Int, max: Int): Option[Int] =
    fields.get(key).flatMap(_.toIntOption).filter(v => v >= min && v <= max)
  private def flag(fields: Map[String, String], key: String): Option[Boolean] =
    fields.get(key).flatMap(_.trim.toLowerCase(java.util.Locale.ROOT) match
      case "true" => Some(true)
      case "false" => Some(false)
      case _ => None)

  def freeu(fields: Map[String, String]): Option[ujson.Obj] =
    fields.get("FreeU Stages").filter(s => s.length <= 4096 && s.count(_ == '[') == 1 && s.count(_ == ']') == 1 &&
      s.count(_ == '{') == 2 && s.count(_ == '}') == 2).filter(_ => fields.get("FreeU Version").contains("2")).flatMap { raw =>
      Try(ujson.read(raw)).toOption.collect { case a: ujson.Arr if a.value.size == 2 => a }.flatMap { stages =>
        def stage(index: Int): Option[(Double, Double)] = stages(index) match
          case o: ujson.Obj if o.value.keySet == Set("backbone_factor", "skip_factor") =>
            def get(k: String, max: Double) = o.value.get(k).collect { case ujson.Num(n) if n.isFinite && n >= 0 && n <= max => n }
            for b <- get("backbone_factor", 2); s <- get("skip_factor", 4) yield (b, s)
          case _ => None
        for a <- stage(0); b <- stage(1) yield ujson.Obj("enabled" -> true, "b1" -> a._1, "b2" -> b._1, "s1" -> a._2, "s2" -> b._2)
      }
    }

  def hypertile(fields: Map[String, String]): Option[ujson.Obj] =
    for
      tile <- integer(fields, "Hypertile tile size", 1, 2048)
      swap <- integer(fields, "Hypertile swap size", 1, 128)
      depth <- integer(fields, "Hypertile max depth", 0, 10)
      scale <- flag(fields, "Hypertile scale depth")
    yield ujson.Obj("enabled" -> true, "tile_size" -> tile, "swap_size" -> swap, "max_depth" -> depth, "scale_depth" -> scale)

  def pag(fields: Map[String, String]): Option[ujson.Obj] =
    for
      scale <- number(fields, "PAG scale", 0, 30)
      adaptive <- number(fields, "PAG adaptive scale", 0, 1)
      layers <- fields.get("PAG applied layers").filter(s => s.nonEmpty && s.length <= 4096)
    yield ujson.Obj("enabled" -> true, "scale" -> scale, "adaptive_scale" -> adaptive, "layers" -> layers)
