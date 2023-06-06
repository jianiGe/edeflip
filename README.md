# Edeflip: Supervised Word Translation between English and Yoruba
## By Ikeoluwa Abioye (Ike.23@dartmouth.edu) and Jiani Ge (Jiani.Ge.23@dartmouth.edu)

## Code
* main changes to MUSE:
  - adapted from [MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE)
  - updated deprecated code
  - removed sentence translation sections as that was not relevant to our project
* analyzed the impact of various embedding types and normalization on the result


## Get monolingual word embeddings
For pre-trained monolingual word embeddings, we highly recommend [fastText Wikipedia embeddings](https://fasttext.cc/docs/en/pretrained-vectors.html), or using [fastText](https://github.com/facebookresearch/fastText) to train your own word embeddings from your corpus.

The data we used can be found here (https://drive.google.com/drive/folders/1ZVLMym3EIjgEzSEVNQBxkKmrJMbWrm6b) with our log files and data.<br>
You can download the English (en) and Yoruba (yo) embeddings this way:
```bash
cd MUSE
# English fastText Wikipedia embeddings
curl -Lo data/wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
# Yoruba fastText Wikipedia embeddings
curl -Lo data/wiki.yo.vec https://drive.google.com/uc?export=download&id=19vfXxahoKDTyNaJoK9grB_i8yvWzfgMj
# Or Yoruba curated FastText embeddings
curl -Lo data/cur.yo.vec https://drive.google.com/uc?export=download&id=13t09-KsbOefIpPEjmbYInimZArtS8lGV

```

## Align monolingual word embeddings
**Supervised**: using a train bilingual dictionary (or identical character strings as anchor points), learn a mapping from the source to the target space using (iterative) [Procrustes](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem) alignment.


### The supervised way: iterative Procrustes (CPU|GPU)
To learn a mapping between the source and the target space, run:
```bash
# for wikipedia Yoruba embeddings
python supervised.py --src_lang en --tgt_lang yo --src_emb data/wiki.en.vec --tgt_emb data/wiki.yo.vec --n_refinement 5 --dico_train default --normalize_embeddings center,renorm --cuda false

# for curated Yoruba embeddings
python supervised.py --src_lang en --tgt_lang yo --src_emb data/wiki.en.vec --tgt_emb data/cur.yo.vec --n_refinement 5 --dico_train default --normalize_embeddings center,renorm --cuda false
```

### Evaluate monolingual or cross-lingual embeddings (CPU|GPU)
We also include a simple script to evaluate the quality of cross-lingual word embeddings on several tasks:

**Cross-lingual**
```bash
python evaluate.py --src_lang en --tgt_lang es --src_emb data/wiki.en-es.en.vec --tgt_emb data/wiki.en-yo.yo.vec --max_vocab 200000 --cuda false --normalize_embeddings center,renorm
```
Reports the precision at top k retrievals.

You can visualize crosslingual nearest neighbors using https://colab.research.google.com/drive/12b6cxewcDWo4MEafPiDwDFWCap918cuy.
