package sdfx.ui.vm

/** Validation of the opt-in partial WD folder workflow. */
object WdBatchUiState:
  def validate(enabled: Boolean, available: Boolean, imagesRoot: String,
               outputName: String): Either[String, Option[(String, String)]] =
    if !enabled then Right(None)
    else if !available then Left("WD v3 batch is unavailable. Configure the local model and tags in Settings and restart the engine.")
    else
      val root = Option(imagesRoot).getOrElse("").trim
      val name = Option(outputName).getOrElse("").trim
      if root.isEmpty then Left("Choose an input image folder for WD v3 batch.")
      else if name.isEmpty || name == "." || name == ".." || name.endsWith(".") || name.exists(c => c == '/' || c == '\\' || c == ':' || c < ' ')
        then Left("Enter a new output folder name without path separators.")
      else Right(Some((root, name)))
