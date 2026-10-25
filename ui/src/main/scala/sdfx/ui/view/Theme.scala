package sdfx.ui.view

import javafx.scene.Scene

/** Applies the app stylesheets (theme colors + shared rules) on top of Modena. */
object Theme:
  private def url(name: String): String =
    val u = getClass.getResource(s"/sdfx/ui/$name")
    if u == null then throw new IllegalStateException(s"missing stylesheet $name") else u.toExternalForm

  def stylesheets(theme: String): Seq[String] =
    Seq(url(if theme == "light" then "light.css" else "dark.css"), url("app.css"))

  def themeClass(theme: String): String = if theme == "light" then "theme-light" else "theme-dark"

  def apply(scene: Scene, theme: String): Unit =
    scene.getStylesheets.setAll(stylesheets(theme)*)
    val root = scene.getRoot
    root.getStyleClass.removeAll("theme-dark", "theme-light")
    root.getStyleClass.add(themeClass(theme))
