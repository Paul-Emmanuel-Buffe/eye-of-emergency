---

# eye-of-emergency

Classification NLP de tweets pour la détection de catastrophes réelles. Projet complet : EDA textuelle, création d'un pipeline de preprocessing sur-mesure, et comparaison de 5 modèles ML (dont un Decision Tree implémenté de zéro). Optimisation par GridSearch axée sur le F1-score.

## Veille Technologique

### Text Mining (Fouille de texte)

Il s'agit de transformer du texte non structuré (des emails, des articles, des tweets, des avis clients) en données structurées et quantifiables qu'un système informatique peut analyser statistiquement.

**Comment ça fonctionne :**

1. **La collecte de données :** On rassemble des textes pertinents provenant de diverses sources (pages web, bases de données internes, réseaux sociaux, formulaires).
2. **Le prétraitement (Nettoyage) :** Formatage du texte. Cela inclut la correction des fautes, le passage en minuscules, le retrait de la ponctuation, la réduction des mots à leur racine (ex: "chevaux" => "cheval"), et la suppression des "mots vides" (ex: le, de, etc.) qui n'apportent pas de sens.
3. **L'exploration et l'analyse :** À cette étape interviennent les algorithmes et le NLP (Traitement du Langage Naturel). Le système va compter la fréquence des mots, repérer la syntaxe, identifier les entités (noms propres, lieux, dates) ou évaluer le ton général du texte.
4. **La restitution (Visualisation) :** Les résultats bruts sont transformés en informations exploitables par l'humain, souvent sous forme de graphiques, de tableaux de bord ou de nuages de mots.

**Quelques applications :** Analyse de sentiment, filtrage automatique, extraction d'informations médicales.

---

### NLP (Natural Language Processing)

Branche de l'IA qui donne aux ordinateurs la capacité de comprendre, d'interpréter et de générer du langage humain de manière intelligente.

Le NLP est un ensemble de techniques qui se divisent en deux grandes catégories :

1. **NLU (Natural Language Understanding) :** Capacité à comprendre. L'ordinateur analyse la grammaire, le contexte, et l'intention. C'est ce qui permet à une machine de comprendre que "Tu noircis le tableau" exprime une accentuation négative d'une situation, et non une action littérale de peinture.
2. **NLG (Natural Language Generation) :** Capacité à produire du texte ou de la parole. Une fois la demande comprise et l'information trouvée, la machine doit formuler une phrase correcte et naturelle pour répondre.

---

### Conclusion : Text Mining vs NLP

**Points communs entre ces deux concepts :**

* **La matière première :** Les deux disciplines travaillent exclusivement sur des données non structurées textuelles.
* **L'automatisation :** Ils traitent à grande échelle des volumes de texte qu'un humain mettrait des années à lire.
* **La préparation des données :** Avant de faire l'un ou l'autre, le texte subit le même "nettoyage" (suppression de la ponctuation, mise en minuscules, découpage en mots ou tokenization).

**Différences fondamentales :**

| Critère | Text Mining | NLP |
| --- | --- | --- |
| **Objectif** | Extraire des données cachées, des modèles et des fréquences. | Comprendre l'intention, le sens et le contexte derrière les mots. |
| **Approche** | Quantitative et statistique (comptage de mots, corrélations). | Qualitative et linguistique (analyse grammaticale, syntaxique et sémantique). |
| **Génération** | Impossible (ne crée pas de nouvelles phrases). | Possible (formule des réponses grâce au NLG). |

> **En résumé :** Si le Text Mining est l'action de recherche d'informations pour en tirer des statistiques ou des mots-clés, le NLP permet à la machine de comprendre le sens et le contexte de ces mots.

---

### Focus NLP : Les sous-domaines

Le NLP est un vaste domaine qui rassemble plusieurs sous-domaines spécialisés et une multitude de techniques.

1. **L'Analyse des sentiments (Opinion Mining) :**
Ce sous-domaine est dédié au décodage des émotions humaines dissimulées au sein d'un texte. Les algorithmes modernes ne se contentent plus d'une simple lecture ; ils sont capables d'évaluer la polarité globale (positive, négative, neutre) tout en détectant des nuances linguistiques complexes telles que l'ironie ou le sarcasme. C'est un outil stratégique incontournable pour les entreprises afin de surveiller leur e-réputation et prévenir les crises.
* **Fonctionnement :** Le texte est nettoyé et converti en vecteurs mathématiques -> Un modèle d'IA analyse ces données chiffrées pour évaluer le contexte global de la phrase -> Le système calcule des probabilités pour attribuer un score de polarité final.
* **Application (Finance et trading algorithmique) :** Des systèmes parcourent des millions de données issues des réseaux sociaux et de la presse financière en temps réel. Dès qu'un sentiment global puissant est détecté sur une entreprise, l'alerte est donnée. Les algorithmes peuvent alors vendre les actions avant une chute boursière.


2. **La Reconnaissance d'Entités Nommées (NER) :**
La NER repère et classe des informations clés (noms, lieux, dates) au sein d'un texte brut.
* **Fonctionnement :** L'algorithme découpe la phrase en segments (tokens) qu'il convertit en vecteurs mathématiques -> Un modèle neuronal évalue rigoureusement le contexte de chaque mot -> Le système calcule des probabilités pour attribuer une étiquette ("Personne", "Lieu") à chaque vecteur -> Cette prédiction transforme une simple chaîne de caractères en données structurées.
* **Application (Tri automatique de CV) :** Le système lit des centaines de CV bruts pour en extraire intelligemment les informations clés (noms, compétences, expériences).


3. **La Traduction Automatique Neuronale (NMT) :**
Ce sous-domaine traduit des phrases entières en capturant leur sens global plutôt que de traduire mot à mot.
* **Fonctionnement :** Le texte source est fragmenté en tokens, transformés en vecteurs. Un réseau "encodeur" extrait une représentation abstraite du contexte. Un second réseau "décodeur", guidé par des mécanismes d'attention, cible les informations pertinentes. Le système calcule enfin des probabilités pour générer séquentiellement la traduction.
* **Application (Sous-titrage en direct) :** Le système capture le contexte global pour générer des sous-titres précis à la volée, permettant à un public mondial de suivre un expert, chacun dans sa langue maternelle.



---

### Les concepts techniques fondamentaux

#### Stop-word (Mot vide)

C'est un mot extrêmement courant, qui pris isolément, n'apporte presque aucune information sur le sens profond de la phrase (ex: le, la, je, et, à, etc.).

En Text Mining, le retrait de ces mots est une étape cruciale du nettoyage :

* **Réduire le bruit :** En filtrant ces mots de liaison, on force l'algorithme à se concentrer uniquement sur les mots porteurs de sens.
* **Améliorer la pertinence statistique :** Sans filtrage, un mot comme "de" arrivera toujours en première position des fréquences, faussant l'analyse.
* **Alléger les calculs :** Les stop-words représentant souvent plus de 50 % du volume d'un texte, les supprimer économise de la mémoire et accélère les algorithmes.

> *Exemple :* "Le petit chat noir dort profondément sur le canapé du salon" -> "petit, chat, noir, dort, profondément, canapé, salon". La machine perd la grammaire, mais conserve 100 % du sens avec deux fois moins de mots.

#### Traitement de la ponctuation et des caractères spéciaux

Le traitement dépend de l'objectif : le Text Mining basique supprime presque tout, tandis que le NLP moderne les conserve souvent pour comprendre le contexte.

* **Ponctuation classique (. ! ?) :** Souvent supprimée pour faciliter le comptage, elle est conservée en NLP pour délimiter les phrases et évaluer l'intensité émotionnelle.
* **Symboles sociaux (@, #) :** Préservés et isolés, car ils identifient des thématiques clés (hashtags) ou des mentions d'utilisateurs.
* **Emojis et devises (😊, €) :** Les émojis sont généralement traduits en texte descriptif (ex: 😊 devient "joie"), et les devises aident à extraire des montants.
* **Bruit informatique (HTML, URLs) :** Les liens et balises sont considérés comme parasites et systématiquement supprimés via des expressions régulières (Regex).

#### Tokens & N-grams

* **Token :** C'est l'unité de base analysée par la machine (généralement un mot ou un sous-mot).
* **N-gram :** C'est une suite de tokens consécutifs (ex: un bigramme regroupe deux mots comme "très chaud"). Cela permet de conserver l'ordre et le contexte.
* **La Tokenisation :** C'est l'opération technique qui découpe le texte brut en tokens, souvent en utilisant les espaces comme séparateurs.

#### Stemming & Lemmatisation

**Le Stemming (Radicalisation) :**
Méthode mécanique et algorithmique. Elle coupe la fin des mots à l'aide de règles heuristiques simples pour ne garder qu'une racine brute (*stem*), qui n'est pas nécessairement un vrai mot (Ex: "mangeait", "mangeur" -> "mang").

* *Over-stemming :* Couper trop loin et confondre des mots différents.
* *Under-stemming :* Ne pas couper assez et rater le lien entre deux mots proches.
* *Quand l'utiliser ?* Lorsque la vitesse est prioritaire, l'application simple, et que la précision orthographique n'est pas critique.

**La Lemmatisation :**
Méthode linguistique et contextuelle. Elle analyse la grammaire et le rôle du mot pour retrouver sa forme canonique du dictionnaire, le *lemme* (infinitif pour les verbes, masculin singulier pour les adjectifs).

* *Quand l'utiliser ?* Lorsque la précision linguistique est indispensable, qu'un affichage lisible par un humain est requis, ou pour des langues complexes (français, allemand, arabe).

#### Vectorisation

Pour qu'un algorithme de Machine Learning puisse analyser du texte, il doit traduire les mots en chiffres. C'est la vectorisation.

1. **Bag of Words (Sac de mots) :**
La méthode la plus simple. Elle ignore la grammaire et l'ordre, se contentant de compter la fréquence du vocabulaire.
* *Le concept :* On crée un dictionnaire de tous les mots uniques du corpus. Pour chaque phrase, on crée un vecteur qui compte les occurrences de chaque mot.
* *La limite :* Donne le même poids à tous les mots. Un mot très fréquent mais inutile prendra trop d'importance.


2. **TF-IDF (Term Frequency-Inverse Document Frequency) :**
Inventé pour corriger le Bag of Words en *pondérant* l'importance des mots. Un mot est jugé important s'il est très présent dans un document précis, mais rare dans le reste du corpus.
* *TF (Term Frequency) :* Fréquence du mot dans le document actuel.
* *IDF (Inverse Document Frequency) :* Rareté du mot dans l'ensemble des documents.
* *Le résultat :* Les mots communs (avec un faible IDF) auront un score proche de 0. Un mot spécialisé (ex: "algorithme") aura un score élevé s'il caractérise le texte, permettant au modèle d'identifier les véritables mots-clés.