# 📄 [ASVspoof2019 vs. ASVspoof5: Assessment and Comparison](https://arxiv.org/abs/2505.15911)


This repository contains the official implementation of the paper ["ASVspoof2019 vs. ASVspoof5: Assessment and Comparison"](https://arxiv.org/abs/2505.15911).

In this work, we conduct a comprehensive evaluation and comparison of two benchmark datasets used developing spoofing countermeasures for automatic speaker verification systems: ASVspoof2019 and ASVspoof5. 

The code has been migrated from Matlab to Python to improve usability and accessibility. If you encounter any issues while using the repo, feel free to sent email [Avishai Weizman](mailto:wavishay@post.bgu.ac.il).
## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch umap-learn confidence_intervals scipy openpyxl 
```

This repository also used on the following repositories:

* [**One-Class Learning Towards Synthetic Voice Spoofing Detection**](https://github.com/yzyouzhang/AIR-ASVspoof):The base code for One Class Softmax (OCS) system. 
* [**Confidence intervals for evaluation in machine learning**](https://github.com/luferrer/ConfidenceIntervals): The repository provides a implementation of the bootstrapping approach to compute confidence intervals for evaluation in machine learning. 

---

## ▶️ How to Run

Follow the instructions below to reproduce key experiments from the paper:

* **UMAP Visualization of Time Embeddings**
  To run the UMAP experiments on time-based embeddings, first [download the required data from Google Drive](https://drive.google.com/file/d/1TTY5BggaaUn4laQr2TmefT_83FoJBaf7/view?usp=drive_link). Then execute:

  ```bash
  python PMF_BASED_Embeddings_Umap/main_umap_2D.py
  ```

* **Histogram Calculation**
  To compute histograms of the embeddings, run:

  ```bash
  python Histogram_Calculations/plot_histograms.py
  ```

* **Distance Calculation Between Histograms**
  To calculate distances between the generated histograms, run:

  ```bash
  python Histogram_Calculations/calculate_distances.py
  ```

* **OCS System Score Evaluation**
  To evaluate the One-Class System (OCS) scores and check the genuine performance, use the following Jupyter notebooks:

  * `OCS_System_Performance/OCS_System_Performance.ipynb`

---

## 📚 Cite The Paper:

If you use this codebase in your research or publications, please consider citing the paper:

```bibtex
@misc{weizman2025asvspoof2019vsasvspoof5assessment,
      title={ASVspoof2019 vs. ASVspoof5: Assessment and Comparison}, 
      author={Avishai Weizman and Yehuda Ben-Shimol and Itshak Lapidot},
      year={2025},
      eprint={2505.15911},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2505.15911}, 
      note = {Accepted to Interspeech 2025},
}
```
---

## 🙌 Acknowledgements

This work is inspired by the ASVspoof Challenge evaluation framework and research in spoofing countermeasures (CM) for automatic speaker verification (ASV) systems.

Thank to the authors of the following repositories:

```bibtex
@ARTICLE{zhang2021one,
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters}, 
  title={One-Class Learning Towards Synthetic Voice Spoofing Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={937-941},
  abstract={Human voices can be used to authenticate the identity of the speaker, but the automatic speaker verification (ASV) systems are vulnerable to voice spoofing attacks, such as impersonation, replay, text-to-speech, and voice conversion. Recently, researchers developed anti-spoofing techniques to improve the reliability of ASV systems against spoofing attacks. However, most methods encounter difficulties in detecting unknown attacks in practical use, which often have different statistical distributions from known attacks. Especially, the fast development of synthetic voice spoofing algorithms is generating increasingly powerful attacks, putting the ASV systems at risk of unseen attacks. In this work, we propose an anti-spoofing system to detect unknown synthetic voice spoofing attacks (i.e., text-to-speech or voice conversion) using one-class learning. The key idea is to compact the bona fide speech representation and inject an angular margin to separate the spoofing attacks in the embedding space. Without resorting to any data augmentation methods, our proposed system achieves an equal error rate (EER) of 2.19% on the evaluation set of ASVspoof 2019 Challenge logical access scenario, outperforming all existing single systems (i.e., those without model ensemble).},
  keywords={},
  doi={10.1109/LSP.2021.3076358},
  ISSN={1558-2361},
  month={},}
```

```bibtex
@software{Confidence_Intervals,
author = {Ferrer, Luciana and Riera, Pablo},
title = {Confidence Intervals for evaluation in machine learning},
url = {https://github.com/luferrer/ConfidenceIntervals}}
```
---

## 📬 Contact

If you have questions, feedback, or want to collaborate, feel free to reach out:
*Avishai Weizman**  
 📧 Email: [Avishai Weizman](mailto:wavishay@post.bgu.ac.il)  

 🔗 GitHub: [github.com/avishai111](https://github.com/avishai111)

 🎓 Google Scholar: [Avishai Weizman](https://scholar.google.com/citations?hl=iw&user=vWlnVpUAAAAJ)  
 
 💼 LinkedIn: [linkedin.com/in/avishai-weizman/](https://www.linkedin.com/in/avishai-weizman/)
 
