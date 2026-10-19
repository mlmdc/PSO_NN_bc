# Breast Cancer Classification with PSO and Neural Networks
# Classification du cancer du sein par PSO et réseaux de neurones

**A MATLAB study comparing Particle Swarm Optimization (PSO) and a feedforward neural network on the Wisconsin Diagnostic Breast Cancer dataset.**  
**Une étude MATLAB comparant l'optimisation par essaim particulaire (PSO) et un réseau de neurones feedforward sur le jeu de données Wisconsin Diagnostic Breast Cancer.**

[← Retour au profil GitHub / Back to GitHub profile](https://github.com/mlmdc) · [Table of contents / Sommaire](#table-of-contents)

![PSO classification output compared with the reference labels](medias/PSO_output.png)

## Table of contents

- [Overview / Présentation](#overview--présentation)
- [Problem and approach / Problématique et approche](#problem-and-approach--problématique-et-approche)
- [Technical decisions / Choix techniques](#technical-decisions--choix-techniques)
- [Results and learnings / Résultats et apprentissages](#results-and-learnings--résultats-et-apprentissages)
- [Getting started](#getting-started)

## Overview / Présentation

This educational project classifies breast mass samples as benign or malignant from 30 features computed from digitized fine-needle aspirate images. It compares a weighted linear classifier whose weights are optimized with PSO against a feedforward neural network. The goal is to explore two classification approaches on the same dataset, not to support clinical decisions.

Ce projet pédagogique classe des échantillons de masses mammaires comme bénins ou malins à partir de 30 caractéristiques calculées sur des images numérisées de ponctions à l'aiguille fine. Il compare un classifieur linéaire pondéré, dont les poids sont optimisés par PSO, à un réseau de neurones feedforward. L'objectif est d'étudier deux approches sur un même jeu de données, et non d'aider à une décision clinique.

**Included / Contenu**

- PSO-based optimization of a linear classifier / Optimisation PSO d'un classifieur linéaire
- Feedforward neural-network training and evaluation / Entraînement et évaluation d'un réseau feedforward
- Output plots and confusion-matrix figures / Graphiques de sortie et matrices de confusion

## Problem and approach / Problématique et approche

The task is to map 30 numeric measurements to one of two diagnoses in the Wisconsin Diagnostic Breast Cancer dataset (569 samples). The implementation uses a thresholded linear combination for PSO and a feedforward network for comparison; both produce binary class labels.

Le défi consiste à associer 30 mesures numériques à l'un de deux diagnostics dans le jeu Wisconsin Diagnostic Breast Cancer (569 échantillons). L'implémentation utilise une combinaison linéaire seuillée optimisée par PSO et un réseau feedforward comme approche de comparaison ; les deux produisent des classes binaires.

## Technical decisions / Choix techniques

**English.** MATLAB was used with its Deep Learning Toolbox (formerly Neural Network Toolbox) and Statistics and Machine Learning Toolbox: `feedforwardnet` trains the network, while `cvpartition` creates folds. PSO searches 30 feature weights using a mean-squared classification error objective. The network uses a linear hidden layer and Levenberg–Marquardt training. This keeps the comparison focused on the optimization method and model structure, but the linear output and threshold are a simple baseline rather than a calibrated probabilistic classifier.

**Français.** MATLAB et ses toolboxes Deep Learning (anciennement Neural Network) et Statistics and Machine Learning fournissent `feedforwardnet` pour entraîner le réseau et `cvpartition` pour créer les partitions. Le PSO recherche 30 poids avec une erreur quadratique moyenne de classification comme objectif. Le réseau utilise une couche cachée linéaire et l'algorithme de Levenberg–Marquardt. Ce choix facilite la comparaison des méthodes, mais la sortie linéaire seuillée reste une approche de base, sans probabilités calibrées.

## Results and learnings / Résultats et apprentissages

The repository contains plots and confusion matrices produced by the experiments. They show that both approaches generate class predictions, but the saved figures are not evidence of performance on an independent test set: the script selects parameters using cross-validation folds and then evaluates predictions over the full dataset. The neural-network fold-selection logic also does not update its recorded best score. Treat the figures as exploratory outputs, not as a validated benchmark. A reliable comparison would require correcting the selection logic and reporting metrics on a held-out test set or with properly nested cross-validation.

Le dépôt contient les graphiques et matrices de confusion issus des expériences. Ils montrent que les deux approches produisent des prédictions, mais ne démontrent pas leurs performances sur un jeu de test indépendant : le script sélectionne les paramètres à partir des partitions de validation croisée, puis évalue les prédictions sur l'ensemble des données. La logique de sélection du réseau ne met pas non plus à jour le meilleur score enregistré. Ces figures sont donc exploratoires et ne constituent pas un benchmark validé. Une comparaison fiable demanderait de corriger cette sélection et de mesurer les résultats sur un jeu de test réservé ou avec une validation croisée imbriquée.

## Getting started

### Requirements

- MATLAB
- The Deep Learning Toolbox (formerly Neural Network Toolbox) and Statistics and Machine Learning Toolbox

### Run

1. Clone this repository and open its directory in MATLAB.
2. Make sure `data.mat` is in the current directory. The script loads it directly.
3. Run:

```matlab
run("lab4.m")
```

The script trains and evaluates the models, prints performance summaries in the MATLAB command window, and opens plots and confusion charts. Results can vary between runs because the optimization and cross-validation use random initialization.

The repository also includes `wdbc.dat`, a dataset description in `opis.txt`, and generated figures in `medias/`.
