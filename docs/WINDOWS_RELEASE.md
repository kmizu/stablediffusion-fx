# Windows v0.02 package

## Use

Extract `SDFX-v0.02-windows-x64.zip` to a writable folder (for example, under your user profile), then run `SDFX\SDFX.exe`. The bundled Java runtime needs no separately installed Java. Keep the extracted directory together: `app`, `runtime`, `python`, and `models` sit next to the executable. The launcher defaults the app home to this extracted SDFX directory, even if the current directory or `SDFX_HOME` points to another checkout. An explicit `--home DIR` overrides that default. Do not run directly inside the ZIP. A writable location is needed for settings, outputs and the Python environment.

Python 3.10–3.12 is needed for the engine; Python and torch are not included. On the first normal launch, the setup dialog can create a virtual environment and download dependencies, or use a Python environment you choose. The torch download is large and uses the CUDA 12.8 wheel index. `SDFX.exe --dry-run` exercises the UI with placeholder generation but still needs a working Python interpreter. NVIDIA GPU and appropriate drivers are recommended for real generation.

To use existing Civitai downloads, place the model files in the corresponding `models` subdirectories. To use an existing reForge installation, open Settings → Model folders → **Add A1111/reForge install...** and select the installation; see the main project README for the supported model folders. Models are not included in the package. Preserve the model creators' respective licences.

`LICENSE`, `THIRD_PARTY_NOTICES.md`, `DISTRIBUTION_NOTICES.md` and the referenced `licenses/*.txt` cover the app and listed dependencies; the bundled runtime's own notices are in `runtime/legal`. This is a portable app-image, not an installer; deleting its extracted folder removes its local data as well.


## WD v3 batch interrogation (partial)

This optional Img2Img workflow tags each input image using a local WD v3 model before generating. It does not implement all backends or options of the original batch interrogator. Models are not bundled or downloaded automatically.

1. In the **same Python environment selected in Settings → Engine**, install the optional ONNX inspection package and ensure ONNX Runtime is installed. For the app-created Windows environment, from the extracted `SDFX` folder run:

   ```powershell
   & .\python\.venv\Scripts\python.exe -m pip install onnx==1.21.0 onnxruntime==1.23.2
   ```

   If Settings specifies a different Python executable, use that executable instead. `onnxruntime==1.23.2` is also in the standard requirements; the explicit command above installs or verifies both packages. Installing them requires network access or locally supplied wheels.
2. Obtain the two files from the fixed official revision of `SmilingWolf/wd-vit-tagger-v3`: [model.onnx](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f/model.onnx) and [selected_tags.csv](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/resolve/7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f/selected_tags.csv). The accepted SHA-256 values are:

   - `model.onnx`: `35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6`
   - `selected_tags.csv`: `298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217`

   These hashes identify the locally reviewed candidate; they have not been independently authenticated against the publisher. In PowerShell, check each file with `Get-FileHash -Algorithm SHA256 <path>`. The worker checks both hashes at startup.
3. Keep both as regular local files in ordinary folders. Hugging Face cache snapshot paths can be symbolic links; copy the target bytes into normal `.onnx` and `.csv` files and select those copies. Paths containing links or junctions are rejected.
4. In **Settings → WD v3 (partial)** select the two files and restart the engine. Reopen Settings to see whether the worker reported WD v3 configured at its last start. Then open **Img2Img → WD v3 batch (partial)**, enable it, choose an image folder and a new output folder name, and submit. Existing output folders are not replaced.

The configured status confirms the files and dependencies passed startup checks; it does not guarantee that ONNX Runtime can load or execute on the machine. Scripts, other extensions, LoRA, ControlNet, masks and hires are unsupported for this batch mode.

## Build the distribution

Build on Windows x64 with Git, PowerShell and a Windows x64 JDK 21. First assemble the final, verified Windows JavaFX fat JAR from the intended source revision (`sbt -batch ui/assembly`); packaging does not build or download anything. Then, from any working directory:

```powershell
& 'path\to\stablediffusion-fx\scripts\package_windows.ps1' -Version '0.02' -JdkHome 'C:\Program Files\Java\jdk-21' -Jar 'path\to\stablediffusion-fx\target\stablediffusion-fx.jar' -OutputRoot 'path\to\release-output'
```

`-Version` defaults to `0.02` and accepts two or three numeric components of one to three digits each; `-JdkHome` defaults to `JAVA_HOME`; `-Jar` defaults to this checkout's `target/stablediffusion-fx.jar`; `-OutputRoot` defaults to `dist/windows-v<Version>` in this checkout. Paths with spaces are supported. Each invocation creates a fresh `build-<UUID>` directory inside the output root and retains its staging files. Its versioned ZIP and `SHA256SUMS.txt` are the release assets. Only tracked Python source/licence files, tracked model README skeletons and tracked `licenses/*.txt` files are copied, along with the requirement files and app and JVM distribution notices; no model weights, venv, settings or outputs are copied. Review the source checkout and JAR before packaging; this script does not verify their provenance.

jpackage creates a classpath app-image and its default linked runtime; the default package version is `0.02`, while the intended release tag and ZIP use `v0.02`. Passing another `-Version` changes the jpackage version, output directory default and ZIP name; it does not modify the Git tag. Refer to [Oracle's JDK 21 packaging overview](https://docs.oracle.com/en/java/javase/21/jpackage/packaging-overview.html). This script does not sign binaries or publish a release.

## Release verification

Check `SHA256SUMS.txt` against the ZIP, inspect its contents and licences, and extract it to a clean writable path containing spaces. From a different working directory, with Java removed from PATH, including tests where the working directory and `SDFX_HOME` point to another valid checkout, run:

```powershell
& 'C:\path with spaces\SDFX\SDFX.exe' --dry-run --screenshot 'C:\path with spaces\smoke.png' --exit-after 15
```

Confirm the process exits successfully and inspect the screenshot; use a working Python for dry-run. Also test the normal first-run setup in an isolated extracted copy and run actual generation with a supported Python environment, GPU and real model. Test model discovery and generation using the user's existing reForge/Civitai files through the native app. Verify the final release source revision, ZIP hash and tag before publication. These are release checks to perform, not claims that they have already passed.
