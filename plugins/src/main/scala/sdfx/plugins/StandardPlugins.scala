package sdfx.plugins

import sdfx.core.plugins.api.{Plugin, XyzAxis}

/** Built-in plugins in a stable display order (scripts first, then extras, then AlwaysOn).
  * docs/PLUGINS.md maps each of them to the reForge built-in it rewrites.
  */
object StandardPlugins:
  def alwaysOn: Seq[Plugin] = Seq(
    new PromptFusionPlugin,
    new ComposableLoraPlugin,
    // guidance
    new FreeUPlugin, new DynamicThresholdingPlugin, new ApgPlugin, new MahiroPlugin, new RescaleCfgPlugin, new PagPlugin,
    new SagPlugin,
    // structure / sampling
    new HyperTilePlugin, new KohyaHrFixPlugin, new MultiDiffusionPlugin, new ModelSamplingPlugin, new LatentModifierPlugin,
    new RandomResolutionsPlugin,
    // image prompts / control
    new ReferencePlugin, new RevisionPlugin, new IpAdapterPlugin, new ControlLLLitePlugin, new PhotoMakerPlugin,
    // inpainting / post
    new SoftInpaintingPlugin, new FooocusInpaintPlugin, new FaceRestorePlugin, new AdetailerPlugin,
    // memory
    new NeverOomPlugin,
    // prompt processing (extension index)
    new RandomizePlugin, new PromptFormatPlugin, new PromptFilterPlugin, new NegativePromptWeightPlugin,
    new NovelAIPromptPlugin, new StyleVariablesPlugin(), new LoraBlockWeightPlugin, new WeightGradientPlugin, new NegPipPlugin,
    new NeutralPromptPlugin, new TokenizerPlugin,
    // sampling (extension index)
    new DetailDaemonPlugin, new VectorscopeCcPlugin, new LatentMirroringPlugin, new AsymmetricTilingPlugin,
    new AntiBurnPlugin, new ReSharpenPlugin, new ColorGradingPlugin, new CfgSchedulePlugin, new CadsPlugin,
    new SaveIntermediatePlugin, new GridNumbersPlugin
  )

  def scripts(extraAxes: () => Seq[XyzAxis]): Seq[Plugin] = Seq(
    new XyzGridPlugin(extraAxes),
    new PromptMatrixPlugin,
    new PromptsFromFilePlugin,
    new LoopbackPlugin,
    new SdUpscalePlugin,
    new OutpaintPlugin,
    new OutpaintMk2Plugin,
    new Img2ImgAltPlugin,
    new LatentUpscalePlugin,
    new SvdPlugin,
    new Zero123Plugin,
    new CustomCodePlugin,
    new UltimateSdUpscalePlugin,
    // checkpoint tools (extension index)
    new ModelMergePlugin, new ModelConverterPlugin,
    // mask-driven img2img (extension index)
    new DepthMaskPlugin, new InpaintBackgroundPlugin, new SeamlessTilePlugin,
    // extension index: prompt and sequence scripts
    new DynamicPromptsPlugin, new OneButtonPromptPlugin, new TestMyPromptPlugin, new ScramblePromptsPlugin,
    new TweakWeightsPlugin, new LoraQueuePlugin, new BatchCheckpointsPromptsPlugin, new SeedTravelPlugin,
    new ShiftAttentionPlugin
  )

  def extras: Seq[Plugin] = Seq(
    new UpscaleExtrasPlugin,
    new FaceRestoreExtrasPlugin,
    new CaptionPlugin,
    new FlippedCopiesPlugin,
    new SplitOversizedPlugin,
    new AutosizedCropPlugin,
    new FocalCropPlugin,
    // image tools (extension index)
    new RemoveBackgroundPlugin, new PersonMaskPlugin, new PixelArtPlugin, new ColorEnhancePlugin, new FsrUpscalePlugin
  )

  /** `extraAxes` lets the host contribute axes of all loaded plugins (including these AlwaysOn ones). */
  def all(extraAxes: () => Seq[XyzAxis]): Seq[Plugin] = scripts(extraAxes) ++ extras ++ alwaysOn

  /** Convenience: all standard plugins with the X/Y/Z grid seeing the standard AlwaysOn axes. */
  def withOwnAxes: Seq[Plugin] =
    lazy val ps: Seq[Plugin] = all(() => ps.flatMap(_.xyzAxes))
    ps
