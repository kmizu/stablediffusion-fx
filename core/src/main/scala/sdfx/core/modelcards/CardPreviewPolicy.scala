package sdfx.core.modelcards

enum CardPreviewMode:
  case Blur, Hide, Show

enum PreviewPresentation:
  case Normal, Blurred, Hidden

/** Naming policy over the literal selected preview URI, never decoded or inferred from a model. */
object CardPreviewPolicy:
  def matches(source: Option[String]): Boolean = source.exists { text =>
    text != null && text.sliding(4).exists { part =>
      part.length == 4 && ascii(part.charAt(0)) == 'n' && ascii(part.charAt(1)) == 's' &&
        ascii(part.charAt(2)) == 'f' && ascii(part.charAt(3)) == 'w'
    }
  }

  private def ascii(c: Char): Char = if c >= 'A' && c <= 'Z' then (c + ('a' - 'A')).toChar else c

  def startup(saved: String): CardPreviewMode = saved match
    case "Blur" => CardPreviewMode.Blur
    case "Show" => CardPreviewMode.Show
    case _      => CardPreviewMode.Hide

  def presentation(source: Option[String], mode: CardPreviewMode, revealed: Boolean): PreviewPresentation =
    if !matches(source) then PreviewPresentation.Normal
    else mode match
      case CardPreviewMode.Hide => PreviewPresentation.Hidden
      case CardPreviewMode.Blur if !revealed => PreviewPresentation.Blurred
      case _ => PreviewPresentation.Normal
