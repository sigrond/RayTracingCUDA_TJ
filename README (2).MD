#BorderRecognition

###Projekt mający na celu znajdować "ramkę" obrazu rozproszeniowego Mie'go.
Projekt jest częścią pracowni dyplmowej inżynierskiej.

"Ramka" jest rysowana funkcją BorderCreation
f(PkX,PkY,PkZ,CCDX,CCDY,CCDZ): R^6 -> R^2^n
funkcja ta nie jest znaną funkcją analityczną i kożysta z ray tracingu

Wstępna wersja projektu zakładała ręczne zaznaczenie punktów na brzegu obrazu rozproszeniowego. 
Po wprowadzeniu tych punktów zostanie zminimalizowana odległość punktów od ramki powstającej za pomocą funkcji 
f (minimalizacja sumy kwadratów odległości) i zostaną uzyskane argumenty funkcji dające ramkę danego obrazu rozproszeniowego. 
Argumenty te opisują położenie kropli zawieszonej w pułapce oraz położenie matrycy CCD względem środka pułapki.




##Atomatyzacja znajdowania szukanych zmiennych

Wstępnie znajduję ramkę (szukane zmienne) przez maksymalizację puntów jasnych wewnątrz ramki (aktualne parametry)
i ciemnych na zewnątrz ramki. Realizuję to za pomocą fminsearch i funkcji celu BrightnesScalarization.
W celu wybrania progu między wartościami ciemnymi i jasnymi za pomocą wyszukiwania binarnego szukam takiej wartości
progowej przy której zadana część punktów jest jasna (~30%). Realizuje to funkcja ThresholdValue.
Wadą tego rozwiązania jest ignorowanie faktycznego położenia przez wybieranie progu na podstawie stosunku jasnych
pixeli do ciemnych, co nie oddaje natury brzegu rozproszeniowego.

W następnym kroku automatyzacji zanajdowania szukanych zmiennych zamierzam w okolicy wcześniej znalezionego brzegu szukać
punktów, które z większą pewnością będą brzegowe. Mając punkty brzegowe wykożystam wcześniej napisaną funkcję celu
MeanSquaredDistance.
Brzeg jest wynikiem dyfrakcji fali płaskiej (światła) z przesłoną. Zatem brzeg (granica między światłem, a cieniem)
jest ukryty pod wzorem dyfrakcyjnym i wyznaczenie go wymaga znalezienia ostatniego minimum i maximum dyfrakcyjnego,
a następnie geometrycznego wyznaczenia granicy swiatła i cienia.
http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/difopa.html#c1
Dane są wybrane punkty ramki (indexy), wyznaczam prostą przechodzącą przez lewego i prawego sąsiada (styczna do ramki), 
znajduję prostą prostopadłą do stycznej przechodzącą przez zadany punkt. Na tej prostej w pewnym otoczeniu (~30px) 
spodziewam się znaleźć wzór dyfrakcyjny. W celu znalezienia pixeli na tej prostej zamierzam skożystać z algorytmu 
Bresenhama. Znajdę największe maximum i minimum (lewo i prwostronna pochodna zerowa, aby pominąć wartość na brzegu odcinka).
Maximum dyfrakcyjne jest w geometrycznym środku między minimum dyfrakcyjnym, a szukaną granicą światła i cienia.

##Raport za semestr 16Z
[Raport](https://github.com/sigrond/BorderRecognition/raw/master/RaportPDI1.odt.pdf)

##Zależności i użyte toolboxy Matlaba
license('inuse')
image_toolbox
matlab
optimization_toolbox

[f,p]=matlab.codetools.requiredFilesAndProducts('BorderRecognition.m')

f = 
…
p = 

1x2 struct array with fields:

    Name
    Version
    ProductNumber
    Certain
	

'MATLAB'	'8.6'	1	true
'Optimization Toolbox'	'7.3'	6	true
'Image Processing Toolbox'	'9.3'	17	true
'Global Optimization Toolbox'	'3.3.2'	90	true



##Wywołanie demonstracji w Matlabie
Demonstarcja z odpowiednimi parametrami wywoływana jest poleceniem:
```Matlab
quickstart
```

##Wyniki profilera
####Wybrane wyniki profilera(RayTracing natywnie w matlabie):

| Function Name		     |	Calls   | Total Time |	Self Time   |
|------------------------|----------|------------|--------------|
| BorderRecognition      |	1 		|  211.103 s |	0.613 s 	|
| BorderFunction 	     |	8360 	|  194.031 s |	5.456 s 	|
| RayTracing 		     |	668800  |  180.999 s |	59.482 s 	|
| fminsearch 		     |	2 		|  105.494 s |	0.208 s 	|
| fminunc 			     |	1 		|	95.173 s |	0.312 s 	|
| RayTracing>findAlpha   |	2657202 |	69.847 s |	40.848 s 	|
| dot 				     |	6659725 |	57.473 s |	57.473 s 	|
| RayTracing>SphereCross |	1328601 |	27.832 s |	17.548 s 	|
| BrightnesScalarization |	293 	|	18.091 s |	2.166 s 	|
| cross 				 |	2657202 |	5.826 s  |	5.826 s 	|

####Wybrane wyniki profilera(RayTracing w MEX C++):

| Function Name		     |	Calls   | Total Time |	Self Time   |
|------------------------|----------|------------|--------------|
| BorderRecognition		 |	1		|  101.503 s |	0.727 s		|
| BorderFunction		 |	11157	|	78.187 s |	69.297 s	|
| fminsearch			 |	2		|	69.403 s |	0.253 s		|
| MeanSquaredDistance	 |	810		|	67.349 s |	0.093 s		|
| fminunc				 |	1		|	19.199 s |	0.273 s		|
| BrightnesScalarization |	310		|	9.121 s	 |	2.403 s		|
| RayTracing_MEX		 |	892560	|	2.998 s	 |	2.998 s		|

####Wybrane wyniki po zmienieniu mało wydajnego kawałka kodu w Border Function:

| Function Name		     |	Calls   | Total Time |	Self Time   |
|------------------------|----------|------------|--------------|
| BorderRecognition		 |	1		|	25.005 s |	0.029 s		|
| fminsearch			 |	2		|	19.839 s |	0.176 s		|
| BorderFunction		 |	11157	|	14.314 s |	6.821 s		|
| MeanSquaredDistance	 |	810		|	12.210 s |	0.087 s		|
| BrightnesScalarization |	310		|	4.791 s	 |	2.275 s		|
| effective_aperture	 |	11157	|	4.567 s	 |	0.472 s		|
| fzero					 |	11157	|	3.986 s	 |	2.847 s		|
| fminunc				 |	1		|	3.354 s	 |	0.013 s		|
| RayTracing_MEX		 |	892560	|	1.717 s	 |	1.717 s		|


##Do zrobienia:
- [x] Mapy Wartości i gradientów - da się zrobić 10x10x10x10x10x10 przy następnym rzędzie wielkości nie starczy pamięci. są mapy dla par wsp. i f-celu
- [x] Wyniki z profilera
- [x] Kolory na obrazie jasnych punktów dla kanałów
- [ ] Liczniki czasu na paskach okienek
- [ ] Optymalizacja dla par wsp. Xi,Xj i f-celu
- [ ] Sprawdzić czy da się ręcznie lepiej ustawić i zapisać wyniki
- [ ] Więcej przykładów
- [x] Wyniki profilera dla RayTracing w C++
- [ ] Fitowanie całek Frenzela
