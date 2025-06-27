# DiT playground

Training logs for this branch available [here.](https://wandb.ai/g4be/osu-mmdit-flow)
Beatmap encoding / decoding taken from [jaswons (Jason Won)](https://github.com/jaswon) [osu-dreamer repository.](https://github.com/jaswon/osu-dreamer/tree/main/osu_dreamer)

# Usage

> [!WARNING]
> Only osu!std (and non lazer) song folders are supported!

Process your available osu maps.

```bash
python3 ./process_maps.py <osu_songs_folder>
```

This should result in a new folder: `data/`. Now just run the training.

```bash
python3 ./train.py data/
```
