**RSNA Knee MRI — a címkesúlyozás és a gradiensakkumuláció javítása**

A javaslat célja, hogy a 0,2-es megbízhatósági súly ténylegesen csökkentse az adott címke hozzájárulását. Ez implementációs specifikáció; a javasolt módosítás pontosságra gyakorolt hatását összehasonlító tanítással kell mérni. Az epochkezelés javítása a felhasználó visszajelzése szerint már elkészült.

**Kiinduló probléma**

A jelenlegi loss az egyes osztályokat külön, mikrobatchenként normalizálja a címkesúlyok összegével. Egyvizsgálatos mikrobatch esetén minden pozitív súly kiesik:

```text
osztályveszteség = súly × BCE / súly = BCE
```

Ezért ugyanazon címke 0,2-es és 1,0-s súlya azonos hozzájárulást eredményez. Nyolc ilyen mikrobatch gradiensének akkumulációja ezt nem javítja meg.

A javasolt megoldás: a súlyozott BCE-t az érvényes címkék darabszámával normalizáljuk, a teljes nyolcvizsgálatos akkumulációs ablakon. A megbízhatósági súly csak a számlálóban szerepeljen. Ehhez a lossfüggvényt és a tanítási ciklust együtt kell módosítani.

**1. Külön célérték, érvényességi maszk és megbízhatósági súly**

| Címke típusa | Célérték `y` | Maszk `m` | Súly `w` |
|---|---:|---:|---:|
| Egyértelmű pozitív címke | 1 | 1 | 1,0 |
| Egyértelmű negatív címke | 0 | 1 | 1,0 |
| Nem említett, kísérletileg gyenge negatívként kezelt | 0 | 1 | 0,2 |
| Bizonytalan / sikertelen címkézés | 0, helykitöltő | 0 | 0 |

A 0,2 veszteségsúly, nem 20%-os pozitív célérték. A gyenge negatív továbbra is `y=0`, de kisebb mértékben befolyásolja a tanítást. A súly nem automatikusan az LLM által közölt bizonyosság: előre meghatározott címkézési szabályhoz rendelt kísérleti érték.

A kizárt címkék célértéke is legyen véges, például nulla: a BCE kiszámítása után a NaN nullával szorzása nem biztonságos maszkolás.

**2. A javasolt célfüggvény**

Legyen `i` az akkumulációs ablak egyik vizsgálata, `c` pedig az egyik célváltozó. Az ablakban osztályonként számoljuk meg az érvényes címkéket:

$$
D_c = \sum_i m_{ic}.
$$

Az adott osztály vesztesége:

$$
L_c = \frac{\sum_i m_{ic}\,w_{ic}\,\operatorname{BCEWithLogits}(z_{ic},y_{ic})}{D_c},\qquad D_c>0.
$$

Legyen az aktív osztályok halmaza:

$$
\mathcal{C}_+ = \{c:D_c>0\}.
$$

A teljes ablak vesztesége:

$$
L = \frac{1}{|\mathcal{C}_+|}\sum_{c\in\mathcal{C}_+} L_c.
$$

**A nevezőben a maszkok összege szerepel, nem a megbízhatósági súlyok összege.** Rögzített maszk és predikció mellett egy címke súlyának `1.0 → 0.2` módosítása pontosan ötödére csökkenti annak loss- és logitgradiens-hozzájárulását. Ez nem jelenti azt, hogy az AdamW teljes paraméterfrissítése is pontosan ötödakkora lesz.

Ehhez elemenkénti BCE szükséges, `reduction="none"` beállítással. A BCEWithLogits bemenete közvetlenül a modell logitja legyen, előzetes sigmoid nélkül. [PyTorch dokumentáció](https://docs.pytorch.org/docs/2.8/generated/torch.nn.BCEWithLogitsLoss.html)

Ez az ablakon belüli osztálynormalizálás nem garantál globálisan egyenlő osztályhozzájárulást az egész epochban. Sok gyenge negatív együttes hatása továbbra is jelentős lehet.

**3. A gradiensakkumuláció módosítása**

A jelenlegi beállítás:

```yaml
train:
  microbatch_studies: 1
  accumulation_steps: 8
```

A tanítási ciklus követelményei:

1. A következő legfeljebb nyolc vizsgálat címkemaszkjaiból előre számoljátok ki a 12 osztály `D_c` értékeit és az aktív osztályok számát. Ugyanazok a vizsgálatok szerepeljenek a nevezőben, amelyeknek a veszteségét az adott ablakban akkumuláljátok.
2. Az ablak elején nullázzátok a gradienseket.
3. A képeket továbbra is egyesével dolgozzátok fel.
4. Minden mikrobatch súlyozott BCE-számlálóit az egész ablakra előre kiszámolt `D_c` nevezőkkel és az egész ablak aktív osztályainak számával normalizáljátok.
5. Minden mikrobatch hozzájárulásából külön történhet `backward()`.
6. Az ablak végén egyszer történjen gradiensvágás és optimizerfrissítés. AMP használatakor a gradiensvágás előtt történjen az unscale; a scheduler az alkalmazott optimizerlépésekhez igazodjon.

Az egy mikrobatchre jutó hozzájárulás tehát:

$$
L_{\text{micro}} = \frac{1}{|\mathcal{C}_+|}\sum_{c\in\mathcal{C}_+}\frac{\sum_{i\in\text{micro}}m_{ic}\,w_{ic}\,\operatorname{BCEWithLogits}(z_{ic},y_{ic})}{D_c}.
$$

**Ezt már nem szabad további nyolccal osztani:** az ablakra normalizálás ezt elvégzi. Az egyes mikrobatch-hozzájárulások összege adja az ablak teljes veszteségét.

Az utolsó, rövidebb ablakhoz a tényleges címkeszámok kellenek. Ha az egész ablakban nincs érvényes címke, hagyjátok ki az optimizer- és schedulerlépést. Ha csak egy mikrobatch üres, annak hozzájárulása nulla; az ablak többi vizsgálata továbbra is tanít.

Nem kell nyolc vizsgálat teljes számítási gráfját egyszerre a GPU-n tartani. A nevezőkhöz előre csak címkemetaadat szükséges; a forward/backward lépések egymás után végrehajthatók.

**4. Az implementáció elfogadási tesztjei**

- Rögzített logits és címkék mellett az egyben számolt nyolcvizsgálatos loss logitgradiense egyezzen a nyolc részletben akkumulálttal. A tesztet először FP32-ben, megfelelő numerikus toleranciával végezzétek. Ez a loss ellenőrzése; a dropouttal vagy BatchNormmal végzett eltérő modell-forwardok nem alkalmasak önmagukban az algebrai egyezés tesztelésére.
- Rögzített maszkok és logits mellett egyetlen címke `1.0 → 0.2` súlyváltása annak hozzájárulását pontosan ötödölje, és a többi címke nevezője ne változzon.
- A kizárt címkék gradiense legyen nulla.
- Az üres mikrobatch, a teljesen üres ablak és a rövidebb utolsó ablak is megfelelően működjön.
- A naplózott optimalizációs loss ugyanazt a célfüggvényt kövesse, mint a backward. Ha külön epochszintű diagnosztikai macro loss is készül más nevezőkkel, annak külön neve legyen.

Ezek elvárt, még az implementáción lefuttatandó tesztek; ez a dokumentum nem állítja, hogy a módosított kód már elkészült vagy átment rajtuk.

**5. Javasolt kísérleti sorrend**

1. Az összehasonlítás alapja az epochkezelés javítását már tartalmazó tanítás legyen.
2. Először csak az új loss és akkumuláció működését próbáljátok ki, a meglévő 0/1 címkesúlyokkal. Maradjon azonos a fold, a címkeállomány, a normalizálás, a modell és a többi hiperparaméter.
3. Csak külön kísérletben kapcsoljátok be a gyenge negatívokat, a részletes címkestátuszok alapján, indokolt célváltozókra.
4. A validáció referenciacímkéi és értékelési maszkjai maradjanak változatlanok a tanítási súlyozás módosításakor. A különböző célfüggvények nyers train loss értékei nem közvetlenül összehasonlíthatók; a rögzített validációs metrikákat kell összevetni.

**6. A címkeadatok szükséges előfeltétele**

A `train_v1.csv` üres mezőiből nem derül ki, hogy „nem említett” vagy „bizonytalan” státuszból származnak. Pusztán az `unmentioned_weight=0.2` konfigurációs beállítással ez a különbség nem állítható helyre.

A gyenge negatív kísérlethez a részletes LLM-kimenetből kell megőrizni a státuszt és a döntés indokát. Az összes üres mező automatikus negatívvá alakítása nem része ennek a javaslatnak. A 0,2-es súly kísérleti kiindulópont, nem igazolt optimum.
