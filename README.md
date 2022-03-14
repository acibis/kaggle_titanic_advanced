# Kaggle Titanic - jak to zrobić lepiej

*CO: takie przejście przez kagglowy zbiór danych [Titanic](https://www.kaggle.com/c/titanic), żeby osiągnąć dobry wynik*

*PO CO:  żebym w końcu miała w jednym miejscu skrót procesu myślowego, który zaszedł w trakcie pracy nad zbiorem. Plus to taki mój mały pamiętniczek, zmusza mnie do myślenia, sprawdzania pojęć, definicji. Więc po co - totalnie dla mnie ;)*

*DLACZEGO PO POLSKU:  bo tutoriali w języku angielskim na temat Titanica jest już milion pięćset sto dziewięćset, a polskich prawie wcale*

## 1. WSTĘP

W mojej [podstawowej wersji](https://github.com/acibis/kaggle_titanic_basic) analizy tego zbioru chodziło mi po prostu o to, żeby wziąć udział w zabawie - zerknąć na dane, przerobić je trochę, wytrenować model, wygenerować wynik i wrzucić na Kaggle. O nabycie praktyki. Tam też jest więcej opisów i definicji podstaw. Teraz chodzi mi o to, żeby mój wynik był lepszy. Oczywiście nie jestem aż takim mózgiem, żeby samej wpaść na wszystkie genialne pomysły - poniższy tekst to wynik researchu internetowego i dokładne śledzenie [pracy tego pana](https://towardsdatascience.com/how-i-got-a-score-of-82-3-and-ended-up-being-in-top-4-of-kaggles-titanic-dataset-bb2875cee6b5), [oraz tego](https://www.kaggle.com/subinium/awesome-visualization-with-titanic-dataset) [i jeszcze tego](https://qiita.com/qualitia_cdev/items/1d9fdf4ee3884af5c3f1). Uczmy się od najlepszych ;)
  
 Wracając do sedna: wciąż ważnych jest tych kilka punktów:

 - model ML przyjmuje jedynie dane liczbowe - jeśli dane mają inną formę (na przykład kolor), należy je przerobić
 - model ML nie przyjmie danych z brakami - jeśli dane mają braki, należy się ich pozbyć (usunąć lub uzupełnić, wedle uznania)
 - jedne cechy mają większy wpływ na przewidywany wynik niż inne - niektórych wcale nie trzeba brać pod uwagę i wrzucać do modelu
 - można tworzyć nowe cechy na podstawie już istniejących. Jak?
 - przed przystąpieniem do tworzenia modelu dobrze jest przeanalizować dane i użyć zdrowego rozsądku oraz intelektu (niestety)
  
  ## 2. PLAN
  1. Wczytać dane i rzucić na nie okiem.
  2. Ogarnąć kontekst historyczno-kulturowy.
  3. Przygotować/wyczyścić dane.
  4. Przeanalizować dane.
  5. Stworzyć nowe cechy, które podniosą skuteczność modelu.
  6. Stworzyć i wytrenować model.
  7. Wgrać wynik do Kaggle.
 
 
  ### 2.1 Wczytać dane i rzucić na nie okiem.
  
  Notatka na przyszłość - zawsze lepiej jest mieć więcej danych, niż mniej. Dlatego połączymy zbiór treningowy z testowym. Najpierw wykonałam analizę na samym zbiorze treningowym i wyniki nie były satysfakcjonujące, więc nie popełniajcie mojego błędu i nie marnujcie czasu.
  
  ```
#import danych treningowych
df_train = pd.read_csv("./train.csv")

#import danych testowych
df_test = pd.read_csv("./test.csv")

# połączenie zbiorów
df = df_train.append(df_test, ignore_index=True, sort=True)
df.head()
 ```
![image](https://user-images.githubusercontent.com/13216011/148648536-4fc0ac60-2971-4f25-94ef-4db089740aef.png)

Tak for fun, czy do naszego datasetu załapał się Leonardo? :D
![image](https://user-images.githubusercontent.com/13216011/149150018-af3c3459-36db-4040-b37a-1311aeda510e.png)
![image](https://user-images.githubusercontent.com/13216011/149150111-2394938d-5e95-4ccc-87aa-079938aef71c.png)

### 2.2 Kontekst historyczny.

Dobrze jest mieć szerszy wgląd w sytuację i dane, które analizujemy. Tutaj ważniejsze fragmenty artykułu z Wikipedii:

>"RMS Titanic – brytyjski transatlantyk typu Olympic, który w nocy z 14 na 15 kwietnia 1912 roku, podczas dziewiczego rejsu na trasie Southampton – Cherbourg – Queenstown – Nowy Jork, zderzył się z górą lodową i zatonął.

>Dane o ofiarach są niejednoznaczne – w zależności od źródeł. Spośród 2208–2228 pasażerów i załogi „Titanica” zginęło ponad 1500 osób. Przeżyło katastrofę tylko około 730. Z pasażerów I klasy zginęło nieco mniej niż połowa, z pasażerów II klasy około 60%, z pasażerów III klasy trzy czwarte. Załogi zginęło prawie 80%.

>W łodziach „Titanica” było miejsce dla ponad 1100 osób, ale wiele z nich było częściowo pustych. Zwłaszcza w pierwszej fazie ewakuacji łodzie odpływały z niewielką liczbą osób. Dopiero w dalszej fazie wypadku łodzie odpływały pełne. Nie podjęto niemal żadnej próby ratowania osób, które znalazły się w wodzie. Jedynie piąty oficer, Harold Lowe, rozdzielił pasażerów ze swej łodzi między inne łodzie i popłynął wydobywać z wody tych, którzy pływali w morzu, ale zrobił to zbyt późno i ocalił tylko kilka osób."

![image](https://user-images.githubusercontent.com/13216011/148606413-b45c3919-2dc8-4183-858f-84bbbe4b1f45.png)

 Po co dorzucam suche fakty z Wiki? Żebyśmy mogli mieć punkt odniesienia. Porównajmy dane z informacjami. Nasz zbiór zawiera 891 wierszy, czyli 40% pasażerów podróżujących Titanikiem. W naszym zbiorze członków brak załogi (informacja potwierdzona, po prostu zbiór ich nie zawiera).
 
  ### 2.3 Brakujące wartości.
  
   Gdzie brakuje nam danych?
  ```
  # brakujące wartości
  df.isnull().sum()
  ```
  ![image](https://user-images.githubusercontent.com/13216011/158076051-09be5c61-8a0b-4faf-854b-636b6802e385.png)

 ##### 2.3.1 Brakujący wiek.
 
 Jak uzupełnić pola w kolumnie wiek? Przede wszystkim sprawdźmy, jak się wiek przedstawia w tej chwili:
 ```
# Sprawdźmy, jak wygląda rozkład wieku w zbiorze
df['Age'].hist(grid=False, color='purple', bins=30)
# max wiek
df['Age'].max()
# min wiek
df['Age'].min()
# z podziałem na grupy wiekowe:
bins= [0, 16, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['0-16','15-20', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
df['Age_label'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False, ordered=True)
df['Age_label'].value_counts().sort_index()
```
![image](https://user-images.githubusercontent.com/13216011/151865702-939eb59f-6437-4418-9880-4aa0bc8d6152.png)

Najwięcej mamy 20stek (24%), potem 30stek(19%), dzieci (poniżej 16 roku życia) stanowią 9%. Chcielibyśmy, żeby nasze uzupełnienia przedstawiały się w miarę podobnie.

**WERSJA 1** - Użyjmy statystyki? Moglibyśmy wpisać do kolumny losowe wartości bazując na średniej wieku i odchyleniu standardowym, to znaczy użyć generatora liczb losowych i jako dolną granicę wieku podać średni wiek minus odchylenie, a jako górną granicę średni wiek plus odchylenie. Tak wyglądałby rezultat:

```
import numpy as np

mean = df["Age"].mean()
std = df["Age"].std()
is_null = df["Age"].isnull().sum()

# compute random numbers between the mean, std and is_null
rand_age = np.random.randint(mean - std, mean + std, size = is_null)
print(rand_age)

# zobaczmy, co nam wyszło i jak to się ma do liczebności grup wiekowych

# pd.Series(rand_age) -> zamień wynik, czyi tablicę numpy na typ pandasowy Series, czyli taka kolumna 
bins= [0, 16, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['0-16','15-20', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
pd.cut(pd.Series(rand_age), bins=bins, labels=labels, right=False, ordered=True).value_counts().sort_index()
```
![image](https://user-images.githubusercontent.com/13216011/150029280-8d13767b-a867-44a5-875c-72ca45d3d64d.png)

Wartości, którymi uzupełniliśmy braki:
```
pd.Series(rand_age).unique()

# array([38, 29, 34, 19, 16, 23, 32, 41, 31, 40, 22, 25, 18, 30, 20, 26, 28,
#        27, 33, 35, 24, 15, 42, 39, 43, 36, 37, 21, 17])
```

Powyżej widać, że zabrakło nam wartości skrajnych, dzieci i osób starszych niż 50 lat (co zresztą jest zgodne z naszą teorią wypełniania średnimi wartościami). Być może takie uzupełnienie będzie wystarczające, a może da się to zrobić lepiej.

**WERSJA 2** - poszukajmy informacji o wieku w innej kolumnie. Każdy z pasażerów posiada przy nazwisku tytuł, zbadajmy tę sprawę:
```
df['title'] = df['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
df['title'].unique()

# znaleźliśmy tytuły:
# Mr - mężczyźni
# Mrs - kobiety meżatki
# Miss - kobiety niezamężne
# Master - chłopiec
# Don - hiszpasńki tytuł arystokratyczny
# Dona - hiszpasńki tytuł arystokratyczny
# Rev - pastor
# Dr - Doktor
# Mme - Madame, kobieta zamężna
# Ms - miss, kobieta niezamężna
# Major - mężczyzna, stopień wojskowy
# Lady - kobieta, tytuł szlachecki
# Sir - mężczyzna, tytuł szlachecki
# Mlle - Mademoiselle, kobieta niezamężna
# Col - mężczyzna, stopień wojskowy
# Capt- mężczyzna, stopień wojskowy
# the Countess - kobieta, tytuł szlachecki
# Jonkheer - nazwisko, błąd w cięciu stringa

# sprowadźmy wszystkie panie i panny do wspólnego mianownika, wrzućmy żołnierzy do jednego worka, tak samo arystokrację
# i Johnkeera (również arystokarata, sprawdźcie wiki)

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Dona": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}


df['Title'] = df['Title'].map(Title_Dictionary)
df['Title'].unique()

# a następnie policzmy dla nich średnią wieku:
titles = df['title'].unique()
means = dict()
for title in titles:
    mean = df.Age[(df["Age"].notnull()) & (df['title'] == title)].mean()
    means[title] = round(mean)
    
means
{'Mr': 32, 'Mrs': 37, 'Miss': 22, 'Master': 5, 'Royalty': 41, 'Officer': 46}

```
Sprawdźmy teraz, jak się ma mediana wieku do tytułów i dorzućmy jeszcze informację o klasie i płci, co zwiększy nam granularność wyników

```
grouped_age_median = df.groupby(['Sex','Pclass','Title'])  \
                        .median() \
                        ['Age']

grouped_age_median
```
![image](https://user-images.githubusercontent.com/13216011/158077346-f2d931df-cd58-4cb9-b8ab-f336e40d36b5.png)

##### 2.3.2 Brakująca cena biletu

Mamy tylko jeden brak w tej kolumnie, więc uzupełnimy go średnią ceną biletu dla klasy 3ciej:
```
df[df['Fare'].isna() == True]
df.loc[df['Fare'].isnull(), 'Fare'] = df.groupby(['Pclass'])['Fare'].mean()[3]
```

##### 2.3.3 Brakujący port zaokrętowania
 
Portu brakuje dla 2 osób w związku z tym nie jest to dużo roboty, żeby sprawdzić jak to było w rzeczywistości używając strony https://www.encyclopedia-titanica.org:
![image](https://user-images.githubusercontent.com/13216011/158078029-a7a1adb1-4d7c-46b8-b558-c54235381c27.png)

```
df.loc[df['Embarked'].isnull(), 'Embarked'] = 'S'
```
 
##### 2.3.3 Brakująca kabina
Możemy zauważyć, że nazwa kabiny składa się z literki i cyfr. Szybki research w internecie podpowie, że litera to oznaczenie podkładu/piętra statku, co jest bardzo istotnym czynnikiem przeżycia, bo z kabin ulokowanych w pewnych miejscach ciężej się przedostać do łodzi ratunkowych. Zerknijcie na schemat:

![image](https://user-images.githubusercontent.com/13216011/150093384-910a9564-ad88-4acb-939d-f961afb4d180.png)

Porzućmy więc litery i zostawmy jedynie oznaczenie pokładu:
```
df['Cabin'].fillna('U', inplace=True) # wypełnijmy puste pola literką U(nknown)
df['Cabin'] = df['Cabin'].apply(lambda x: x[0]) # obetnijmy pierwszą literę z każdego pola
df['Cabin'].unique() #spawdźmy, co dostaliśmy

#array(['U', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T'], dtype=object)

df['Cabin'].value_counts()

# U    687
# C     59
# B     47
# D     33
# E     32
# A     15
# F     13
# G      4
# T      1
```

### 3. Czyścimy dane i przerabiamy na liczbowe

Najpierw wyczyśmy/przeróbmy te kolumny, który istnieją w naszym zbiorze.

Po pierwsze, wszystkie kolumny, w których mamy wartości nieliczbowe, musimy zamienić na liczbowe, i mamy na to 2 sposoby. Albo integer encoding, w którym przypiszemy słowom cyfry ( jeśli nasz zbiór zawiera na przykład zwierzęta, kota, psa i myszojelenia, to umawiamy się, ze kot = 1, pies = 2 a myszojeleń = 3). I niby sprawa załatwiona, bo komputer już rozumie, że to 3 różne stworzenia, ale przecież 3 jest większe od 1, więc może wymyślić jakieś nieistniejące związki. Dlatego mamy metodę numer 2, OneHot Encoding (this is one hot name :P), czyi stowrzenie tylu nowych kolumn, ile unikalnych wartości posiadamy, i wpisania 1 w tę jedną, która się zgadza, a 0 w resztę. Czyi w powyższym przypadku tworzymy kolumny kot, pies, myszojeleń, i jeśli coś jest psem, to dostaje 1 w kolumnie pies i 0 wszędzie indziej.
  
```
# tworzymy nowe kolumny dla tytułów i usuwamy kolumne Title
titles_dummies = pd.get_dummies(df['Title'], prefix='Title')
df = pd.concat([df, titles_dummies], axis=1)
df.drop('Title', axis=1, inplace=True, errors='Ignore')

# tworzymy nowe kolumny dla Klasy i usuwamy kolumne Pclass
titles_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, titles_dummies], axis=1)
df.drop('Pclass', axis=1, inplace=True, errors='Ignore')

# tworzymy nowe kolumny dla etykietek wieku i usuwamy kolumne Age_label
titles_dummies = pd.get_dummies(df['Age_label'], prefix='Age_label')
df = pd.concat([df, titles_dummies], axis=1)
df.drop('Age_label', axis=1, inplace=True, errors='Ignore')

# tworzymy nowe kolumny dla portu zaokrętowania i usuwamy kolumne Embarked
titles_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, titles_dummies], axis=1)
df.drop('Embarked', axis=1, inplace=True, errors='Ignore')

# tworzymy nowe kolumny dla kabiny i usuwamy kolumne Cabin
titles_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')
df = pd.concat([df, titles_dummies], axis=1)
df.drop('Cabin', axis=1, inplace=True, errors='Ignore')

```

Po drugie, kolumny z 2 tylko wartościami zamieniamy na liczbowe używając wartości binarnych:

```
#zamieniamy wartości w kolmnie Sex na 0 i 1
df['Sex'] = df['Sex'].map({'male':1, 'female':0})
```

Po trzecie usuwamy niepotrzebne już kolumny:

```
# usuwamy kolumnę Age z brakami, bo stworzyliśmy Age_combined
df.drop('Age', axis=1, inplace=True, errors='Ignore')

# usuwamy kolumnę Name, już się nie przyda
df.drop('Name', axis=1, inplace=True, errors='Ignore')

# usuwamy kolumnę PassengerId, nie przyda się
df.drop('PassengerId', axis=1, inplace=True, errors='Ignore')
```

Została nam jeszcze kolumna Ticket, do której wcześniej nie zaglądaliśmy. Unikalnych wartości jest tam 929, trochę dużo, rzucmy więc na nie pobieżnie okiem i sprawdźmy, co mówi [encyklopedia](https://www.encyclopedia-titanica.org/cabins.html).

![image](https://user-images.githubusercontent.com/13216011/158078889-b135deb9-3e70-4efa-a86b-009ad517f8c6.png)

Widać, że tickety mają prefixy, spróbujmy wyłuskać tę informację.

Teraz zerknijmy na kolumny, które zawierają informację o towarzyszach podróży: SibSp i Parch, czyli liczba dzieci/rodziców i rodzeństwa/współmałżonków. Możemy na ich dodstawie stworzyć nową kolumnę, która określa wielkość jednej podróżującej ze sobą rodziny.

```
df['Family_size'] = df['Parch'] + df['SibSp'] + 1
```

Następnie stwórzmy dodatkowe kolumny, które określają wielkość rodziny

Sigle : pasażer podróżujący samotnie = 1
Small_family :  2 <= x <= 4 osoby
Large_family : więcej niż 5 osób
```
df['Sigle'] = df['Family_size'].map(lambda s: 1 if s == 1 else 0)
df['Small_family'] = df['Family_size'].map(lambda s: 1 if 2 <= s <= 4 else 0)
df['Large_family'] = df['Family_size'].map(lambda s: 1 if 5 <= s else 0)
```


Wow, w końcu, czas na analizę danych! Często wykresy mówią więcej niż obliczenia (przypomnijcie sobie badanie brakujących danych sprzed 10 minut), zwizualizujmy więc nasze dane z uwzględnieniem kolumny, która najbardziej nas interesuje, czyli Survived. Kod użyty do wykonania wykresów jest w notebooku, tutaj wrzucam tylko rezultat.

![image](https://user-images.githubusercontent.com/13216011/150317660-53cc038d-24b4-44bf-985f-9e3f1972e9a5.png)

![image](https://user-images.githubusercontent.com/13216011/150324050-980cf671-798c-49d7-a8cf-fe0460f63228.png)

![image](https://user-images.githubusercontent.com/13216011/150329129-fdc5fffc-0202-4618-9475-ab818daefa33.png)



