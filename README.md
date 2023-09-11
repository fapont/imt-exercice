# 🎯 Objectif

Création d'une application web permettant de prédire le prix d'un bien immobilier à partir de certaines caractéristiques. L'objectif de ce TP est de vous familiariser avec les problématiques de développement et de déploiement d'un modèle de Machine Learning. Ce TP se déroule en 2 parties:

1. Création d'un modèle de Machine Learning permettant de prédire le prix d'un bien immobilier
2. Création d'une API permettant de faire appel à ce modèle

# 🗃️ Ressources

Pour réaliser ce TP nous avons besoin d'avoir :

- [Python](https://www.python.org/downloads/) installé ainsi que de pouvoir travailler avec un Jupyter notebook
- Un IDE (integrated development environment) afin de pouvoir éditer les fichiers de code. Nous utiliseronw [VsCode](https://code.visualstudio.com/), un éditeur de code développé par Microsoft, afin de faciliter ce travail

De plus nous aurons besoin d'être familier avec les librairires python suivantes :

- [`Scikit-Learn`](https://scikit-learn.org/stable/) : utiliser des modèles de Machine Learning
- [`Pickle`](https://docs.python.org/3/library/pickle.html) : sérialiser des objets Python (transformation d'une structure Python en un fichier pouvant être stocké puis reconstruit)
- [`Pandas`](https://pandas.pydata.org/) : manipulation de DataFrame (structure de données)
- [`Flask`](https://flask.palletsprojects.com/en/2.2.x/) : développement d'application web idéalement conçu pour des APIs

# ➕ Installation

## 1. Connexion à l'environnement de travail

Afin de faciliter la réalisation du TP, un environnement de travail vous est fourni. Suivez les indications suivantes :

- Ouvrez le lien suivant https://onyxia.dev.heka.ai/
- Sur le menu de gauche cliquez sur "Mon Compte"
- Entre le nom d'utilisateur et le mot de passe qui vous a été fourni et cliquez sur "Connexion"

Vous êtes maintenant connecté à Onyxia ! Il s'agit d'un datalab développé par l'INSEE permettant de facilement lancer des applications.

Cliquez ensuite sur "Catalogue de service". Une page apparait avec les services principaux, l'un d'entre eux se nome "Vscode-python". Cliquez sur "Lancer".

![Alt text](img/image-2.png)

Modifier le nom du service dans la case "Nom personalisé" puis cliquez sur "Lancer".

![Alt text](img/image-1.png)

Copier le mot de passe affiché, cliquez sur le lien présent dans le fenêtre. Vous êtes maintenant redirigé vers le VsCode en ligne, entrez le mot de passe copié et cliquez sur "Submit". \
Vous êtes maintenant connecté à votre environnement et près à commencer le TP !

_Note_:

- le lancement de l'environnement peut prendre quelques minutes, ne vous inquiétez pas si vous ne voyez pas l'interface de VsCode tout de suite.
- pour garantir le bon fonctionnement du TP ne modifiez pas les paramètres de lancement de l'instance VSCoe (CPU, RAM, ...)

## 2. Set up du TP

**Toutes les commandes suivantes sont à exécuter dans un terminal**

Afin de lancer un terminal, selectionnez le menu sur la barre latérale gauche puis "Terminal" -> "New Terminal".

![Alt text](img/image-3.png)

Ensuite, effectuez les actions suivantes :

1. Cloner le repository du TP en utilisant la commande suivante:

```bash
git clone git@github.com:fapont/imt-exercice.git
```

Cela va créer un dossier `imt-exercice` contenant tous les fichiers nécessaires pour ce TP.

2. Vérifier que Python est correctement installé

```bash
python3 --version
```

3. Installer les dépendances nécessaires pour ce TP. Optionnel pour les étudiants utilisant les instances VSCode (vous devriez avoir une erreur sur en lançant cette commande depuis les environnements virtuels que l'on vous a fourni).

```bash
cd imt-exercice # Se placer dans le dossier du TP
pip install -r requirements.txt
```

_Note_: `pip` est un outil permettant d'installer des packages Python.

4. Tout est prêt ! Vous pouvez ouvrir le fichier `main.ipynb` et commencer le TP.
   Le fichier `main.ipynb` est un Jupyter Notebook. C'est un outil très pratique pour faire du développement itératif. Il permet d'exécuter du code Python par bloc et de visualiser les résultats directement dans le notebook.

# 📊 Données

Les données utilisées pour notre TP sont issues du jeu de données publique des [valeurs foncières françaises de 2021](https://www.data.gouv.fr/en/datasets/demandes-de-valeurs-foncieres/). Ces données ont été retravaillées pour vous et se trouvent dans le fichier `data.csv`.

# 🧑‍💻 Déroulement du TP

## Partie 1: entraînement d'un modèle de Machine Learning

L'objectif de cette partie est de créer un modèle de Machine Learning permettant de prédire le prix d'un bien immobilier à partir de certaines caractéristiques. Une fois ce modèle entraîné nous pourrons le sauvegarder et l'utiliser dans notre application web (partie 2).

Ouvrez le fichier `main.ipynb` et suivez les instructions pour créer votre modèle.

## Partie 2: création d'une webapp

Afin de développer le backend de notre application web nous utilisons le [framework Flask](https://flask.palletsprojects.com/en/2.2.x/). L'objectif de cette partie est de comprendre le fonctionnement des serveurs Web et de s'en servir pour appeler notre modèle et faire de l'inférence grâce à notre modèle. Afin de communiquer avec cette dernière nous utiliserons le protocole `HTTP`, nous vous recommandons de lire cet article expliquant les différentes requêtes possibles (GET, POST, ...) et leur application: https://www.ionos.fr/digitalguide/hebergement/aspects-techniques/requete-http/.

1. Lancez le fichier `app.py` avec la commande `python3 app.py` depuis votre terminal. Votre serveur web va se lancer et être accessible via des requêtes HTTP. Vérifiez que la connexion à votre serveur fonctionne: `curl http://localhost:5678/hello`, ou directement depuis votre navigateur `https://single-project-<random-id>.lab.dev.heka.ai/proxy/5678/hello` à partir des instances VSCode. **Pour la suite du TP on notre `<your-url>` pour remplacer soit `http://localhost:5678` soit `https://single-project-<random-id>.lab.dev.heka.ai/proxy/5678`**.

2. Dans le dossier `templates` vous trouverez un fichier _index.html_ qui contient un code HTML un peu plus complexe qu'un simple Hello World. Créer une nouvelle route `/app` permettant de renvoyer à l'utilisateur le contenu de la page _index.html_. Il est possible d'utiliser la fonction [`render_template`](https://flask.palletsprojects.com/en/2.0.x/quickstart/#rendering-templates) de Flask pour cela.
   Vérifier que votre code fonctionne en vous rendant à l'adresse suivante sur votre navigateur: `<your-url>/app`.

3. Créer une nouvelle route `/predict` permettant à l'utilisateur de passer des données via un [`form`](https://www.w3schools.com/html/html_forms.asp) (doit-on utiliser une requête POST ou GET ?). Cette route devra effectuer dans l'ordre:

- **Lecture des données**: nous transitons l'information de notre client vers notre backend via un formulaire. Flask permet de récupérer ces données dans le corps de la fonction grâce à [l'objet request](https://www.digitalocean.com/community/tutorials/processing-incoming-request-data-in-flask). Habituellement les formulaires sont utilisés pour les applications Web mais dans le cas de simples APIs il est préférable d'utiliser les formats JSON/XML/Protobuf.
- **Vérification de la donnée**: est-ce que la donnée contient tous les champs que l'on souhaite (house_type, nb_room, ...) ? Dans le cas où la donnée est mal formattée, renvoyer un [code d'erreur 400](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/400).
- **Chargement du modèle**: après avoir sauvegardé votre modèle au format `pickle`, il est temps de le recréer et d'y faire appel comme à la fin de la partie 1.
- **Transformation des données & Prédiction**: à partir de notre objet `Pipeline` recréé vous pouvez appliquer la transformation de données et prédire le prix de la maison.
- **Renvoi du résultat**: retournez la valeur prédite ainsi que le [status 200](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/200).

4. Vérifiez que votre nouvelle route fonctionne:

```bash
curl -X POST -d "postcode=75&house_type=Maison&house_surface=130&nb_room=5&garden_area=300"  localhost:5678/predict
```

Ou bien testez directement depuis la page web en remplissant le formulaire !

**Notes**

- `curl` est un outil nous servant ici de client `HTTP` permettant d'envoyer des requêtes. Les requêtes peuvent être de plusieurs types, par exemple GET pour récupérer des données et POST pour en envoyer.
- Comme vous pouvez le remarquer lorsque l'on utilise la commande `curl`, la donnée reçue est `<h1>Hello world</h1>`. Lorsque l'on ouvre notre navigateur on peut y voir apparaitre un magnifique **Hello world** formaté. Votre navigateur vous permet de formatter le HTML que vous recevez à l'écran mais ce dernier n'est rien de plus qu'un client HTTP aggrémenté de fonctions d'affichages.

## Partie 3: Pour aller plus loin

Cette section vous propose d'explorer différents sujets auxquels des ingénieurs dans le data sont confrontés. L'objectif est de vous exposer ces problématiques et libre à vous de creuser les sujets qui vous intéressent les plus.

### Transformation de données avancée (Simple)

Dans notre modèle nous avons transformé les variables catégorielles en variables numériques. Cependant dans beaucoup de modèles il est aussi nécessaire d'appliquer des transformations aux variables numériques. Par exemple, dans le cas de la régression linéaire, il est nécessaire de centrer et réduire les variables numériques ou bien de les normaliser. Vous pouvez rajouter à votre `Pipeline` une étape de scaling des variables numériques. Cela devrait améliorer les performances de votre régression linéaire mais pas forcément celles de votre modèle de forêt aléatoire. Essayez de comprendre pourquoi. 😉

### Intégration du code postal (Intermédiaire & Avancé)

L'objectif est d'utiliser le `Code Postal` afin d'améliorer les performance de notre modèle. Il y a plusieurs approches possibles:

1. (Intermédiaire) Utiliser la variables `Code Postal` comme une variable catégorielle. Cela permettra à notre modèle de prendre en compte le code postal à condition de choisir le bon encodage (pour rappel il n'existe pas de relation d'ordre entre les codes postaux). Vous pouvez peut être réutiliser le OneHotEncoder de `Scikit-Learn` pour cela (combien de colonnes va comporter notre jeu de données après ça ? A quoi ressemblerait notre modèle linéaire ?)

2. (Avancé) Enrichir notre jeu de données en incluant des données externes. Vous pouvez par exemple utiliser le [dataset](https://www.insee.fr/fr/statistiques/4265429?sommaire=4265511) fourni par l'INSEE. Il contient le nombre d'habitants par commune. Vous pouvez utiliser ces données pour enrichir votre jeu de données d'entrainement. Par exemple, si une maison se trouve dans une commune de 1000 habitants, alors on peut rajouter une colonne `nb_habitant` avec la valeur `1000`. Vous pouvez aussi utiliser ces données pour créer une nouvelle variable catégorielle `taille_commune` qui prendra les valeurs `petite`, `moyenne` ou `grande` en fonction du nombre d'habitants de la commune. Cet enrichissement devrait pouvoir améliorer les performances de votre modèle.

_Note_:

- toutes les transformations que vous effectuerez sur votre jeu de données d'entraînement devront être effectuées lors de la prédiction, don réfléchissez bien à ce que vous faites comme transformations.
- si vous souhaitez intégrer cette nouvelle variable dans votre appplication Web il vous faudra modifier le fichier `templates/index.html` en vous inspirant de ce qui est déjà fait. Un peu de HTML ça ne fait pas de mal 😃

### Docker (Avancé)

Aujourd'hui notre application nécessite Python et plusieurs dépendances pour fonctionner correctement. Dans un contexte de production il est courant de ne pas savoir exactement sur quelle machine tourne son programme (ex: Kubernetes choisi parmi un cluster de machine). Il est impensable de devoir installer toutes les dépendances de tous nos programmes sur toutes les machines ! C'est pour répondre à cette problématique que [Docker](https://www.docker.com/) a été créé. Il permet de créer des images qui vont empaquêter toutes les dépendances et le code nécessaire pour faire fonctionner notre application. L'objectif est de créer une image Docker permettant de faire fonctionner notre application web correctement. Vous pouvez vous inspirer de [ce tutoriel](https://www.digitalocean.com/community/tutorials/how-to-build-and-deploy-a-flask-application-using-docker-on-ubuntu-20-04).

### MLOps (Avancé)

Lorsqu'on cherche à améliorer notre modèle il est courant d'augmenter la taille de notre jeu de de données d'entrainement, de rechercher les meilleurs hyperparamètres, de repenser l'architecture, ... De fait, on ne peut pas se permettre de repasser par la phase exploratoire dans un Jupyter Notebook, de réentrainer notre modèle, de le packager et de remplacer le modèle existant par le nouveau. En pratique toutes ces étapes sont automatisées grâce à des pipelines spécifiques. Cette automatisation rentre dans le scope du `MLOps` (Machine Learning Operations). On vous invite à vous renseigner sur ces principes grâce à cet excellent site https://ml-ops.org/. Cette discipline est à cheval entre plusieurs métier: `Data Scientist`, `Data Engineer`, `Software Engineer` et `DevOps`.

Afin de vous initier à cette discipline vous pouvez développer une nouvelle route `/retrain/<data-file>/<model>` permettant de re-entrainer votre modèle à partir de nouvelles données. Cette route devra effectuer dans l'ordre. Voici les objectifs de cette route:

- **Lecture des données**: à partir de la variable `data-file` fournie par l'utilisateur, lire les nouvelles données d'entrainement. Cette donnée est supposée se trouver dans le repertoire courant de l'application web (ex: `data.csv`).

- **Choix du modèle**: à partir de la variable `model` fournie par l'utilisateur, choisir le type de modèle à entraîner. Par exemple si `model=linear` alors on entraîne un modèle de régression linéaire, si `model=forest` alors on entraîne un modèle de forêt aléatoire, etc...

- **Entrainement du modèle**: à partir du modèle choisi, entraîner le modèle sur les nouvelles données. Sauvegarder le modèle au format `pickle` dans le repertoire courant de l'application web (ex: `model.pkl`). Ou alors si on veut pouvoir conserver un historique de nos modèles, on peut sauvegarder le modèle dans un repertoire `models` avec un nom unique contenant le timestamp d'entraînement (ex: `models/model-<timestamp>.pkl`).

- **Modification de la route `/predict`**: rajoute des arguments permettant de choisir le modèle à utiliser pour la prédiction
