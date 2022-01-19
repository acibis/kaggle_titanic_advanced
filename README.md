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
  
  ```
# import danych
df = pd.read_csv("./train.csv")
# przykładowe 3 wiersze
data.sample(3)
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
  
   ![image](https://user-images.githubusercontent.com/13216011/148649032-b081e9d1-d8f2-44d8-9109-75064b0aa64c.png)

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
![image](https://user-images.githubusercontent.com/13216011/150015296-7cd6dd88-c932-4ced-bfbc-b9b41c793f14.png)

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
# Mr - mężczyzna
# Mrs - kobieta meżatka
# Miss - kobieta niezamężna
# Master - chłopiec
# Don -  tytuł grzecznościowy używany w Hiszpani
# Rev - pastor
# Dr - doktor
# Mme - madame, kobieta zamężna
# Ms - miss, kobieta niezamężna
# Major - mężczyzna, stopień wojskowy
# Lady - kobieta, tytuł szlachecki
# Sir - mężczyzna, tytuł szlachecki
# Mlle - mademoiselle, kobieta niezamężna
# Col - mężczyzna, stopień wojskowy
# Capt- mężczyzna, stopień wojskowy
# the Countess - kobieta, tytuł szlachecki
# Jonkheer - nazwisko, błąd w cięciu stringa

# sprowadźmy wszystkie panie i panny do wspólnego mianownika
df.loc[df.title == 'Mlle', 'title'] = 'Miss'
df.loc[df.title == 'Mme', 'title'] = 'Mrs'
df.loc[df.title == 'Ms', 'title'] = 'Miss'

# a następnie policzmy dla nich średnią wieku:
titles = df['title'].unique()
means = dict()
for title in titles:
    mean = df.Age[(df["Age"].notnull()) & (df['title'] == title)].mean()
    means[title] = round(mean)
    
means
#  'Mr': 32,
#  'Mrs': 36,
#  'Miss': 22,
#  'Master': 5,
#  'Don': 40,
#  'Rev': 43,
#  'Dr': 42,
#  'Major': 48,
#  'Lady': 48,
#  'Sir': 49,
#  'Col': 58,
#  'Capt': 70,
#  'the Countess': 33,
#  'Jonkheer': 38
```

Jeżeli zaglądniemy do naszych brakujących danych i sprawdzimy, jakie tytuły mają osoby, które nie posiadają informacji o wieku, to dowiemy się, że są to:
```
nulls = df[df['Age'].isnull()].copy()
nulls.groupby('title')['Name'].count()
# Dr          1
# Master      4
# Miss       36
# Mr        119
# Mrs        17
```

Brakuje nam wieku jednego doktora, 4 chłopców, i tak dalej. Kiedy będziemy chcieli uzupełnić brakujący wiek na podstawie tytułu osoby:
```
nulls = df[df['Age'].isnull()].copy()
for index,row in nulls.iterrows():
    nulls.loc[index, 'Age'] = means[row['title']]
bins= [0, 16, 20, 30, 40, 50, 60, 70, 80, 90]
labels = ['0-16','16-20', '20s', '30s', '40s', '50s', '60s', '70s', '80s']
pd.cut(nulls['Age'], bins=bins, labels=labels, right=False, ordered=True).value_counts().sort_index()

# 0-16       4
# 16-20      0
# 20s       36
# 30s      136
# 40s        1
# 50s        0
# 60s        0
# 70s        0
# 80s        0
```
Wartości, którymi uzupełniliśmy braki:
```
nulls['Age'].unique()

# array([32., 36., 22.,  5., 42.])
```
Panie (Mrs) i panowie (Mr) są najliczniejszą grupą, uzupełniliśmy też wiek 4 chłopców. To, czego nam wciąż brakuje, to informacja o dziewczynkach. Ponieważ nie mamy rozróżnienia między dorosłą kobietą a dziewczynką tak, jak między dorosłymi mężczyznami a chłopcami (Mr vs Master), pozostaje nam się pogodzić z tym, że dziewczynki wiekowo pomijamy.

**WERSJA 3** - poszukajmy informacji o wieku w innych kolumnach i połączmy je ze sobą. Oprócz informacji o tytule, mamy też informację o klasie, którą podróżował pasażer:

```
g = sns.catplot(y="Age",x="title",hue="Pclass", data=df, kind="bar", size = 8)
g.fig.set_figwidth(12)
g.fig.set_figheight(6)
```
![image](https://user-images.githubusercontent.com/13216011/150096095-221fa8cb-0cb1-471a-a551-4fbb34e0e2c5.png)

Wiemy, że brakuje nam 1 doktora, 4 chłopców, 17 pań, 36 panien i wielu mężczyzn, zobaczmy więc, jak się przedstawiają liczbowo propozycje mediany wieku dla nich na podstawie klasy:

```
df[df['title'].isin(['Mr', 'Mrs', 'Miss', 'Master', 'Dr'])].groupby(["title", "Pclass"])["Age"].median()
```
![image](https://user-images.githubusercontent.com/13216011/150106837-f1e5c5b2-daaf-4a3e-86df-628f90f32c45.png)

Wartości, którymi uzupełnilismy braki:
```
nulls['Age'].unique()

# array([26. , 31. , 18. , 40. ,  4. , 24. , 30. , 46.5])
```
Nie wiem który ze sposobów jest najlepszy, więc stworzymy 3 kolumny i będziemy je testować w modelu.
![image](https://user-images.githubusercontent.com/13216011/150029850-75eff009-06ae-4c47-9e94-72f7c0e0283a.png)

```
# wiek na podstawie mediany
df['Age_median'] = df['Age'].copy()
df.loc[df['Age_median'].isnull(), 'Age_median'] = rand_age

# wiek na podstawie tytułu
df['Age_title'] = df['Age'].copy()
df['Age_title'].fillna(-1, inplace=True)

for index, row in df.iterrows():
    if row['Age_title'] == -1:
        df.loc[index, 'Age_title'] = title_medians[row['title']]

# wiek na podstawie tytułu i klasy
df['Age_combined'] = df['Age'].copy()
combined_medians = df[df['title'].isin(['Mr', 'Mrs', 'Miss', 'Master', 'Dr'])].groupby(["title", "Pclass"])["Age"].median()

for key,value in dict(combined_medians).items():
    df.loc[(df['Age_combined'].isnull()) & (df['title']==key[0]) & (df['Pclass']==key[1]),'Age_combined'] = value
```

Przy okazji zobaczmy, jak różne wyniki w kolumnie wiek jest otrzymaliśmy:

![image](https://user-images.githubusercontent.com/13216011/150111680-db2b561f-6e60-4711-b3c5-e678ce7a38f5.png)

To może sugerować, że jednak wypełnianie wieku medianą przyniesie słabe rezultaty:

![image](https://user-images.githubusercontent.com/13216011/150112890-638f8e5e-9022-4229-9204-c4b9c2e477fc.png)



Mnie już mózg paruje, więc wrzucam mema na rozluźnienie:

![image](https://user-images.githubusercontent.com/13216011/150030698-7fcb8f76-8cae-4adf-85df-b56c211c31a2.png)

 ##### 2.3.2 Brakująca kabina.
 
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

Ok, ready? Spróbujemy przewidzieć pokład na podstawie połączonych mocy klasy(Pclass), portu(Embarked) i ceny(Fare). Na początek rzućmy okiem na związek klasy z pokładem:
```import matplotlib as mpl
plt.style.use('ggplot')

g = df.groupby('Cabin')['Pclass'].value_counts(normalize=True).unstack()
g.plot(kind='bar', stacked='True', figsize=(10,4) ).legend(bbox_to_anchor=(1, 0.5), title="Pclass")
```
![image](https://user-images.githubusercontent.com/13216011/150032796-24f851a7-396a-430f-bd4b-f4bfee35ff2d.png)

A następnie na związek klasy i ceny biletu z pokładem:
```
import seaborn as sns
g = sns.catplot(y='Fare', x='Cabin', hue='Pclass', data=df.sort_values(by=['Cabin']), kind='bar', height=10, aspect=2)
g.fig.set_figwidth(12)
g.fig.set_figheight(4)
```

![image](https://user-images.githubusercontent.com/13216011/150113391-53e33d7b-59e1-4333-9646-6e53c2212df1.png)

Taki sam wykres można stworzyć dla zależności podkład - cena - port. Można na ich podstawie wnioskować, że uda nam się uzupełnić braki w rodzaju kabiny po połączeniu wszystkich tych składników. Przede wszystkim pogrupujmy ładnie dane:
```
df.groupby(['Pclass', 'Embarked', 'Cabin'])['Fare'].mean()

Pclass  Embarked  Cabin
# 1       C         A         38.357743
#                   B        145.964018
#                   C         98.582533
#                   D         85.586000
#                   E         92.905840
#                   U        102.376526
#         Q         C         90.000000
#         S         A         40.731763
#                   B         85.372283
#                   C        101.630442
#                   D         49.719906
#                   E         46.448750
#                   T         35.500000
#                   U         53.751986
# 2       C         D         13.333350
#                   U         26.961667
#         Q         E         12.350000
#                   U         12.350000
#         S         D         13.000000
#                   E         11.333333
#                   F         23.750000
#                   U         20.421854
# 3       C         F         22.358300
#                   U         11.042634
#         Q         F          7.750000
#                   U         11.231751
#         S         E         11.000000
#                   F          7.650000
#                   G         13.581250
#                   U         14.749523
```

Powyższa lista ładnie obrazuje to, czego szukaliśmy, ale do wypełniania braków potrzebna nam inna struktura:
klasa , port, cena -> numer kabiny
Musimy więc przekształcić nasz słownik:

```
# słownik jak wyżej
cabins_dict = dict(df.groupby(['Pclass', 'Embarked', 'Cabin'])['Fare'].mean())


# nowy słownik gdzie klucz: Klasa, Port, Cena, a wartość: Kabina
formatted_cabins_dict = dict()

for k,v in cabins_dict.items():
    new_key = (k[0], k[1], v)
    new_value = k[2]
    
    formatted_cabins_dict[new_key] = new_value
    
# usuwamy ze słownika literkę U(nknown), nią nie chcemy uzupełniac braków
ready_cabins = dict()
for k,v in formatted_cabins_dict.items():
    if v != 'U':
        ready_cabins[k] = v
```

Uzupełniamy brakujące dane:

```
# najpierw 2 funkcje, które pomogą dopasować Fare do odpowiedniego klucza

def find_closest(alls, ele):
    
    smallest = 100000
    
    for a in alls:
        if abs(a - ele) < smallest:
            smallest = abs(a - ele)
            found = a
            
    return(a)    

def find_fare(pclass, embarked, fare):
    
    all_fares = []
    for k,v in ready_cabins.items():
        if k[0] == pclass and k[1] == embarked:
            all_fares.append(k[2])
            
    if len(all_fares) > 0:
        found_fare = find_closest(all_fares, fare)
        return(ready_cabins[(pclass, embarked, found_fare)])
    else:
        return('U')      
```

```
# nowa kolumna na uzupełnione wartości
df['Cabin_combined'] = df['Cabin'].copy()
df['Cabin_combined'] = df.apply(lambda x: find_cabin(x.Cabin_combined, x.Pclass, x.Embarked, x.Fare), axis=1)
```

