# 画像生成 AI 入門：Python による拡散モデルの理論と実践

リサーチサイエンティスト [北田 俊輔](https://shunk031.me/), Ph.D.

[![deploy-book](https://github.com/shunk031/coloso-python-diffusion-models/actions/workflows/deploy.yaml/badge.svg)](https://github.com/shunk031/coloso-python-diffusion-models/actions/workflows/deploy.yaml)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/LICENSE)
![Python](https://img.shields.io/badge/🐍%20Python-3.9+-orange)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.16.0-orange)](https://github.com/huggingface/diffusers)

<p align="center">
 <a href="https://www.youtube.com/watch?v=-IPEUOcPTas">
    <img width="560" src="https://raw.githubusercontent.com/shunk031/coloso-python-diffusion-models/main/assets/coloso_kitada_video.png">
  </a>
</p>

本レポジトリでは [Coloso （コロソ）](https://coloso.jp/) で開講されている [`"画像生成 AI 入門：Python による拡散モデルの理論と実践"`](https://coloso.jp/programming/researchscientist-kitada-jp) で扱う notebook を管理しています。各 notebook は Jupyter notebook としてまとめられており、Google Colab で実行されることを想定しています。

[Coloso](https://coloso.jp/event/creativecoloso_jp) は **「業界トップクラスの専門家のノウハウをオンラインで学ぶ」** ことを目標に作られた VOD （ビデオ・オン・デマンド）　型オンライン教育サービスです。

## 🤗 講座の中心は "拡散モデル"

<p align="center">
  <img width="560" src="https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202306/012516-831/jp-researchscientist-kitada-example03.gif">
</p>

現在注目されているテキストから画像を生成するモデルは、denoising diffusion probabilistic model (DDPM) [[Ho+ NeruIPS'20]](https://arxiv.org/abs/2006.11239) と呼ばれる、ノイズ除去拡散確率モデルを元にしています。

これまでとは異なる新たな生成モデルとしてより高精度な画像の生成が可能で従来の手法として主流であった Generative Adversarial Network (GAN) [[Goodfellow+ NeurIPS'14]](https://arxiv.org/abs/1406.2661) を超える性能を持っています。
複雑なデータ分布でも学習可能でその分布を解析的に評価することができます。

この講座は、画像生成 AI の主役となる拡散モデルの理解と実践を通し最新の画像生成技術について学びたい方におすすめです！

## 🎓 講座の内容

- **講座の紹介**: 画像生成に関する AI の基礎から、拡散モデルの理論、そして Python を使った実践的なアプローチまでをリサーチサイエンティストの北田俊輔が段階的にお教えします。
- **無制限視聴**: 一回の購入で、リサーチサイエンティスト 北田俊輔が教える、拡散モデルの理論と実践を盛り込んだ講義動画 30 本を期間制限なしで受講することができます。

## 📄 実習資料

| Section | Lecture | Colab | GitHub |
|---------|---------|-------|--------|
| 1: Introduction to Diffusion Models and Stable Diffusion | 1: Welcome! | | |
| | 2: Overview of the Course | | |
| | 3: Play with Stable Diffusion! | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-01-03.ipynb) | [lectures/section-01-03.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-01-03.ipynb) |
| 2: Basic Knowledge of Deep Learning | 4: About Deep Learning (1) | | |
| | 5: About Deep Learning (2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-01-03.ipynb) | [lectures/section-02-05.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-02-05.ipynb) |
| | 6: The Transformer Model | | |
| 3: Basics of Diffusion Model | 7: Overview of Generative Model | | |
|  | 8: Score-based Generative Model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-08.ipynb) | [lectures/section-03-08.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-08.ipynb) |
| | 9: Denoising Diffusion Probabilistic Model (1) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-09.ipynb) | [lectures/section-03-09.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-09.ipynb) |
| | 10: Denoising Diffusion Probabilistic Model (2) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-10.ipynb) | [lectures/section-03-10.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-10.ipynb) |
| | 11: Beyond Conventional GANs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-11.ipynb) | [lectures/section-03-11.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-03-11.ipynb) |
| 4: Key Researches Based on Non-Diffusion Models | 12: About CLIP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-04-12.ipynb) | [lectures/section-04-12.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-04-12.ipynb) |
| | 13: Overview of Non-Diffusion Models (1) | |
| | 14: Overview of Non-Diffusion Models (2) | |
| 5: Key Researches Based on Diffusion Models | 15: Overview of Diffusion Models (1) | | |
| | 16. Overview of Diffusion Models (2) | | |
| 6: Latent Diffusion and Stable Diffusion | 17: Overview of Latent Diffusion and Stable Diffusion | | |
| | 18: Components of Stable Diffusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-06-18.ipynb) | [lectures/section-06-18.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-06-18.ipynb)
| 7: Play with Diffusion Models | 19: Stable Diffusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-19.ipynb) | [lectures/section-07-19.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-19.ipynb)
| | 20: Textual Inversion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-20.ipynb) | [lectures/section-07-20.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-20.ipynb) |
| | 21: DreamBooth | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-21.ipynb) | [lectures/section-07-21.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-21.ipynb) |
| | 22: Attend-and-Excite | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-22.ipynb) | [lectures/section-07-22.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-22.ipynb) |
| | 23: ControlNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-23.ipynb) | [lectures/section-07-23.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-23.ipynb) |
| | 24: Prompt-to-Prompt | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-24.ipynb) | [lectures/section-07-24.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-24.ipynb) |
| | 25: InstructPix2Pix | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-25.ipynb) | [lectures/section-07-25.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-25.ipynb) |
| | 26: unCLIP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-26.ipynb) | [lectures/section-07-26.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-26.ipynb) |
| | 27: Paint-by-example | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-27.ipynb) | [lectures/section-07-27.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-27.ipynb) |
| | 28: LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-28.ipynb) | [lectures/section-07-28.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-28.ipynb) |
| | 29: Safe Latent Diffusion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-29.ipynb) | [lectures/section-07-29.ipynb](https://github.com/shunk031/coloso-python-diffusion-models/blob/main/lectures/section-07-29.ipynb) |
| 8: Bonus Track | 30: Summary of the entire course and future prospects of image generation AI | | |

## ❓ 疑問点・修正点

疑問点や修正点は本レポジトリの [issue](https://github.com/shunk031/coloso-python-diffusion-models/issues) にて管理しています。不明点などがございましたら以下を確認し、解決方法が見つからない場合は新しく issue を作成してください。

- https://github.com/shunk031/coloso-python-diffusion-models/issues

## 🔗 関連リンク

- 講座ページ | リサーチサイエンティスト 北田俊輔 | Coloso. | コロソ。https://coloso.jp/programming/researchscientist-kitada-jp 
- Colaboratory へようこそ - Colaboratory https://colab.research.google.com
- Huggingface Diffusers https://huggingface.co/docs/diffusers/index 

## License

MIT
