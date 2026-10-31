package sdfx.engine

import sdfx.core.config.Settings
import sdfx.core.plugins.api.ChoiceSource
import sdfx.core.protocol.FileEntry

class ModelKeywordCatalogSuite extends munit.FunSuite:
  test("embedding choices and path resolution use the loaded catalog") {
    val home = FakeEngine.tempHome()
    val services = new AppServices(home, Settings(), dryRun = true, builtins = Nil,
      persistSettings = false, installShutdownHook = false)
    try
      services.catalog.set(CatalogSnapshot(embeddings = Seq(FileEntry("folder/name", "D:/models/folder/name.pt"))))
      assertEquals(services.catalog.choices(ChoiceSource.Embeddings), Seq("folder/name"))
      assertEquals(services.catalog.pathOf(ChoiceSource.Embeddings, "folder/name"), Some("D:/models/folder/name.pt"))
    finally
      services.shutdown()
      FakeEngine.deleteTree(home.root)
  }
