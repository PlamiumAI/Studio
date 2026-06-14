# PlamiumAI Studio — Installation Guide (Beta)
# Notice: It is not guaranteed that the steps in this guide will even work. If you experience any errors, which you likely will, feedback will be very appreciated. You can report anything in our Discord server: https://discord.gg/tBSYJQQyR
PlamiumAI Studio is a Flask web application for generating AI music. It can produce full tracks with two different engines (MusicGen and ACE-Step), create album cover art with Stable Diffusion, separate songs into stems, and help you design songs through an AI chat assistant powered by Google Gemini, Cerebras, or a local Ollama model.

This guide walks you through getting it running from scratch.

## What you'll need

You should have Python 3.10 or 3.11 installed. Some of the machine-learning dependencies are slow to support newer versions, so those two are the safest choices. The application will use a CUDA GPU automatically if one is available, but it falls back to the CPU otherwise, so a GPU is helpful but not required. Plan for at least 8 GB of RAM (16 GB or more is more comfortable), and keep several gigabytes of disk space free for the model checkpoints, which are fairly large.
# Notice: The owner and developer of this project DOES NOT have a CUDA GPU. This program has been only tested on CPU, meaning we don't know if it will work properly for those who do have a GPU. Report any bugs to v2_0s on Discord.
# First, pull this project:
```bash
git clone https://github.com/PlamiumAI/Studio.git plamiumai_studio && cd plamiumai_studio
```
## Setting up the project folder

The script expects the two music engines to live as subfolders right next to it, because on startup it adds `./audiocraft` and `./ACE-Step-1.5` to the Python path. Your finished layout should look like this:

```
PlamiumAI/
├── app.py                ← the main script
├── audiocraft/           ← the MusicGen engine (you'll clone this)
├── ACE-Step-1.5/         ← the ACE-Step engine (you'll clone this)
│   └── checkpoints/      ← ACE-Step model weights go here
├── templates/
│   └── index.html        ← the web interface
└── exports/              ← created automatically on first run
```

## Creating a virtual environment

It's best to keep everything in an isolated environment so the dependencies don't clash with other projects on your machine:

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

## Installing the core dependencies

These packages are needed no matter which features you use, so install them first:

```bash
pip install flask python-dotenv requests numpy scipy soundfile torch torchaudio
```

If you have a CUDA-capable GPU and want hardware acceleration, install the GPU build of PyTorch from pytorch.org rather than the plain `torch torchaudio` shown above. The standard install works fine on CPU as well.

## Installing the music engines

The application is built around two separate generation engines, and you'll want both. MusicGen is required for the app to even start, while ACE-Step powers the default generation mode and the stem-separation feature.

### Audiocraft (MusicGen)

MusicGen comes from Meta's Audiocraft library. Because the script imports it the moment it launches, the application will not run at all without it. Clone the repository into your project folder and install it in editable mode:

```bash
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .
cd ..
```

The first time you actually generate a MusicGen track, the model weights (for example `facebook/musicgen-small`) are downloaded from Hugging Face automatically and cached, so that initial generation will take a little longer than later ones.

### ACE-Step 1.5

ACE-Step is the default engine the app reaches for when you generate music, and it's also what performs stem separation, so installing it is strongly recommended. Clone it into a folder named exactly `ACE-Step-1.5` beside the script, then install it:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git ACE-Step-1.5
cd ACE-Step-1.5
pip install -e .
cd ..
```

After installing the library, you need to download the ACE-Step model checkpoints and place them inside `ACE-Step-1.5/checkpoints/`. The application looks for the turbo configuration and a small language model in that directory, so the weights must be present before ACE-Step generation will work.

If ACE-Step isn't installed, the app will still launch and MusicGen will work, but anything that relies on ACE-Step (the default generation mode and stem extraction) will report that it's unavailable. You'll see a confirmation in the console either way: a success message if the library was found, or a note explaining it couldn't be imported.

## Installing cover art generation (optional)

Album cover art is generated with Stable Diffusion through the `diffusers` library, and this feature is turned on by default. To enable it, install:

```bash
pip install diffusers transformers accelerate pillow
```

If you skip this, the application runs normally and simply leaves covers blank instead of generating them.

## Choosing your AI assistant backend

# In the future, I plan to make this more user-friendly, but currently, the app is in beta.
Near the top of the script there's a single setting that controls which chat backend the songwriting assistant uses:

```python
AI_MODE = 3   # 3 = Google Gemini, 1 = Cerebras, 0 = local Ollama
```

The default is Google Gemini. Whichever you pick, you'll configure it through a `.env` file placed beside the script.

For **Google Gemini** (mode 3), create a `.env` file containing your key:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash
```

The variable `GOOGLE_API_KEY` works as an alternative name if you already have one set.

For **Cerebras** (mode 1), provide your key instead:

```
CEREBRAS_API_KEY=your_key_here
```

For **Ollama** (mode 0), you run a model locally. Install Ollama, make sure it's running on its default port, and pull the model the app expects:

```bash
ollama pull phi4-mini
```

If you'd like the optional Discord Rich Presence feature, you can also add a `DISCORD_CLIENT_ID` line to your `.env`, though it stays off until you enable it in the app's settings.

## Running the application

Once everything is in place, start the server:

```bash
python app.py
```

Then open **http://localhost:5000** in your browser. On first launch the app creates its `exports` folders for finished tracks, cover images, stems, and uploaded vocals.

One thing to keep in mind: the default launch runs Flask in debug mode and listens on all network interfaces, which is fine for local use but not something you'd want exposed directly to the public internet.

## A few common snags

If the app refuses to start with an import error mentioning audiocraft, that library either isn't installed or isn't sitting in a folder beside the script. If generation works for MusicGen but ACE-Step features report being unavailable, the ACE-Step library or its `checkpoints` folder is missing. If covers never appear, the `diffusers` and `pillow` packages probably aren't installed, or cover generation has been switched off in the settings. And if the chat assistant returns errors, double-check that the API key in your `.env` matches the backend selected by `AI_MODE`.
