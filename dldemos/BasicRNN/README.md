1. Download [IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

2. Modify the directory in `read_imdb`.

3. Run `main.py` to train and test the language model. You can:

- Use `rnn1` or `rnn2`
- Switch the dataset by modifying `is_vocab` parameter of `get_dataloader_and_max_length`
- Tune the hyperparameters

to do more experiments.
