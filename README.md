# Handwritten Text OCR

## Descrizione

Handwritten Text OCR è un sistema per il riconoscimento di caratteri, parole e brevi testi scritti a mano.

Il progetto combina tecniche di elaborazione delle immagini, una rete neurale convoluzionale e modelli linguistici. L'immagine viene prima suddivisa in righe, parole e caratteri; ogni carattere viene classificato dalla rete neurale e le lettere ottenute vengono successivamente controllate usando dizionari italiani e inglesi e statistiche sul contesto.

Il sistema distingue le 26 lettere dell'alfabeto latino e restituisce sia il testo letto direttamente dalla rete sia il testo finale dopo l'eventuale correzione.

## Obiettivo

Una rete che riconosce ogni carattere separatamente può commettere errori tra lettere graficamente simili. Per questo motivo il progetto non si limita alla classificazione delle immagini, ma utilizza più livelli di analisi:

- struttura geometrica dell'immagine;
- probabilità prodotte dalla CNN;
- alternative OCR per ogni carattere;
- frequenza delle parole nei dizionari;
- lingua più probabile;
- relazione tra parole vicine nella stessa frase.

L'obiettivo è ottenere un riconoscimento più affidabile senza sostituire automaticamente parole che la rete ha riconosciuto con elevata sicurezza.

## Flusso generale

Immagine manoscritta
        |
        v
Segmentazione in righe, parole e caratteri
        |
        v
Preprocessing di ogni carattere
        |
        v
Classificazione con la CNN
        |
        v
Generazione di più candidati OCR
        |
        v
Rilevamento della lingua
        |
        v
Controllo ortografico
        |
        v
Valutazione del contesto tra parole vicine
        |
        v
Testo finale e immagine annotata

Il coordinamento dell'intera pipeline è affidato a `TextRecognizer`, definito in `src/text_recognizer.py`.

## Segmentazione dell'immagine

La segmentazione, implementata in `src/segment_word.py`, individua le parti dell'immagine che contengono inchiostro e produce un insieme ordinato di riquadri.

L'elaborazione comprende:

1. conversione dell'immagine in scala di grigi;
2. riduzione del rumore;
3. binarizzazione tramite soglia di Otsu;
4. rimozione dei bordi dell'immagine;
5. operazioni morfologiche per rendere più continui i tratti;
6. ricerca delle componenti connesse;
7. eliminazione delle componenti troppo piccole;
8. unione di punti e segni diacritici con il carattere principale;
9. separazione delle componenti troppo larghe, che possono contenere lettere collegate;
10. raggruppamento dei caratteri prima in righe e poi in parole.

Ogni carattere è rappresentato da un oggetto `CharacterSegment`, che contiene le coordinate del riquadro e un'indicazione di eventuale incertezza della segmentazione.

## Preprocessing dei caratteri

Prima di essere fornito alla rete neurale, ogni segmento viene trasformato da `src/preprocessing.py`.

Il preprocessing:

- converte l'immagine in scala di grigi;
- aumenta il contrasto tra scrittura e sfondo;
- inverte i colori quando lo sfondo è chiaro;
- isola i pixel appartenenti al carattere;
- elimina lo spazio vuoto esterno;
- può rimuovere piccoli collegamenti laterali rimasti dalla segmentazione;
- ridimensiona il carattere conservandone le proporzioni;
- centra il risultato in un'immagine nera di 28 x 28 pixel;
- converte l'immagine in un tensore normalizzato nell'intervallo `[-1, 1]`.

Il tensore finale ha forma `(1, 1, 28, 28)`: batch, canale in scala di grigi, altezza e larghezza.

## Rete neurale

La rete `CharacterNN`, definita in `src/model.py`, è una CNN progettata per classificare 26 lettere.

La parte convoluzionale contiene:

- una convoluzione da 1 a 32 mappe di caratteristiche;
- attivazione ReLU e max pooling;
- una convoluzione da 32 a 64 mappe di caratteristiche;
- una seconda attivazione ReLU e un secondo max pooling.

L'immagine iniziale di 28 x 28 pixel viene trasformata in 64 mappe da 7 x 7. Queste caratteristiche vengono appiattite e passate a un classificatore composto da un livello da 128 neuroni, ReLU, dropout e un livello finale da 26 valori.

L'uscita della rete contiene logits. La funzione softmax li converte nelle probabilità associate alle lettere da `a` a `z`.

## Costruzione delle parole

Per ogni carattere non viene conservata soltanto la lettera più probabile, ma anche le migliori alternative prodotte dalla rete.

La prima scelta di ogni posizione forma la parola OCR grezza. Le alternative vengono combinate da `build_ocr_candidates()` mediante beam search: dopo ogni carattere vengono mantenute solo le combinazioni con probabilità complessiva maggiore.

Questo passaggio consente al correttore di recuperare una parola valida anche quando una lettera corretta si trova al secondo o terzo posto tra le previsioni della CNN.

## Rilevamento della lingua

`src/language_detection.py` confronta la parola e i candidati OCR con i dizionari italiano e inglese.

Per ogni lingua viene considerato:

- se la parola compare esattamente nel dizionario;
- se uno dei candidati OCR è una parola valida;
- il punteggio delle eventuali correzioni;
- la frequenza della parola.

Il sistema seleziona l'analisi con le prove linguistiche più forti. Una parola presente in entrambi i dizionari può essere segnalata come compatibile con entrambe le lingue.

## Correzione ortografica

La correzione è implementata in `src/spell_correct.py` e utilizza SymSpell insieme ai dizionari di frequenza.

I suggerimenti vengono ordinati considerando:

- probabilità OCR;
- distanza di modifica dalla parola riconosciuta;
- frequenza nel dizionario;
- somiglianza visiva e ordine dei caratteri.

Il correttore distingue tre situazioni principali:

- parola riconosciuta come corretta;
- possibile errore con una correzione affidabile;
- parola sconosciuta per la quale non esiste una correzione abbastanza sicura.

Le parole brevi o ambigue vengono trattate in modo prudente per evitare sostituzioni arbitrarie.

## Analisi contestuale

La correttezza di una parola isolata non garantisce che sia quella giusta nella frase. `src/context_scoring.py` applica quindi una seconda valutazione alle parole incerte.

I modelli caricati da `src/ngram_language_model.py` contengono frequenze di coppie di parole consecutive, chiamate bigrammi. Per esempio, una coppia molto frequente riceve un punteggio maggiore di una combinazione rara o assente.

La correzione contestuale viene applicata soltanto quando:

- il testo contiene più parole;
- è possibile individuare una lingua dominante;
- la parola contiene caratteri incerti o probabilità non elevate;
- esiste un'alternativa molto vicina alla parola originale;
- l'alternativa riceve prove sufficienti dai vicini della stessa riga.

Le parole riconosciute con alta confidenza vengono normalmente conservate.

## Risultati prodotti

Il riconoscimento restituisce strutture dati a diversi livelli.

`CharacterPrediction` contiene lettera prevista, probabilità, alternative, coordinate e posizione del carattere.

`WordRecognitionResult` contiene parola OCR, parola finale, lingua rilevata, confidenza media, stato ortografico e suggerimenti.

`TextRecognitionResult` raccoglie il testo completo, il numero di righe, i risultati delle singole parole e il percorso dell'immagine annotata.

L'immagine annotata mostra i riquadri dei caratteri e distingue visivamente le previsioni incerte da quelle accettate.

## Interfaccia grafica

`src/qt_app.py` fornisce un'interfaccia PySide6 per selezionare un'immagine e visualizzare:

- testo OCR grezzo;
- testo finale;
- lingua rilevata;
- confidenza media;
- stato della correzione;
- suggerimenti alternativi;
- dettagli delle previsioni dei singoli caratteri;
- immagine con i segmenti annotati.

L'interfaccia utilizza il checkpoint ottenuto dal fine-tuning locale quando disponibile; in caso contrario utilizza il modello addestrato su EMNIST.

## Addestramento iniziale

`src/train.py` addestra la CNN sul dataset supervisionato EMNIST Letters.

`src/dataset.py` corregge l'orientamento originale delle immagini EMNIST, converte le etichette dall'intervallo `1-26` all'intervallo `0-25`, normalizza le immagini e costruisce i DataLoader per training, validation e test.

Durante ogni epoca vengono calcolate cross-entropy loss e accuracy. Il checkpoint viene aggiornato soltanto quando migliora l'accuracy sul validation set.

Il risultato è `models/best_ch_cnn.pt`, che contiene pesi del modello, stato dell'ottimizzatore, epoca e accuracy di validazione.

## Fine-tuning supervisionato

`src/finetune_local_words.py` continua l'addestramento partendo dal modello EMNIST e lo adatta alle immagini locali.

Le etichette non vengono ricavate dal nome del file. Sono lette dal manifest `manifests/images_words.csv`, che associa il percorso dell'immagine alla trascrizione corretta:

```csv
image,expected_text
sample_001.png,casa
sample_002.png,gatto
```

Per ogni parola lo script verifica che il numero di segmenti coincida con il numero di lettere della trascrizione. Le immagini incompatibili vengono escluse, evitando di associare automaticamente lettere a segmenti non affidabili.

Durante il fine-tuning vengono applicati piccoli spostamenti e rumore casuale ai caratteri. Il checkpoint risultante è `models/best_ch_cnn_local_words.pt`.

Il caricamento di una nuova immagine nell'applicazione non provoca addestramento: il modello esegue soltanto inferenza. Per modificare i pesi è necessario disporre della trascrizione corretta ed eseguire esplicitamente il fine-tuning.

## Valutazione

Il progetto contiene due livelli di valutazione.

`src/evaluate.py` valuta la CNN sul test set EMNIST e produce accuracy complessiva, accuracy per lettera, matrice di confusione, confusioni più comuni ed esempi classificati in modo errato.

`src/evaluate_words.py` valuta l'intera pipeline usando un manifest di immagini annotate. Misura:

- accuratezza della segmentazione;
- accuratezza dei caratteri tramite distanza di Levenshtein;
- accuratezza delle parole OCR;
- accuratezza del testo finale;
- precision, recall e F1 nel rilevamento degli errori ortografici;
- presenza della correzione attesa al primo posto o tra i primi tre suggerimenti.

Gli errori vengono classificati come errori di segmentazione, OCR, testo finale, rilevamento ortografico, correzione ortografica o esecuzione.

## Dati e modelli

Il progetto utilizza:

- EMNIST Letters per l'addestramento iniziale;
- immagini locali annotate per il fine-tuning;
- dizionari di frequenza italiani e inglesi;
- dizionari di bigrammi italiani e inglesi.

I principali file generati o utilizzati sono:

```text
models/best_ch_cnn.pt
models/best_ch_cnn_local_words.pt
data/dictionaries/it_frequency.txt
data/dictionaries/en_frequency.txt
data/dictionaries/it_bigrams.txt
data/dictionaries/en_bigrams.txt
manifests/images_words.csv
```

`src/build_bigram_dictionary.py` permette di costruire un dizionario compatto di bigrammi a partire da un corpus di frasi in formato Tatoeba TSV o TSV.BZ2.

## Organizzazione dei moduli

| Modulo | Responsabilità |
| --- | --- |
| `alphabet.py` | Definizione delle 26 classi di output |
| `dataset.py` | Caricamento e preparazione di EMNIST |
| `model.py` | Definizione della CNN |
| `train.py` | Addestramento iniziale |
| `finetune_local_words.py` | Fine-tuning supervisionato sulle immagini locali |
| `evaluate.py` | Valutazione della CNN sui caratteri EMNIST |
| `evaluate_words.py` | Valutazione dell'intera pipeline sulle parole |
| `segment_word.py` | Segmentazione in righe, parole e caratteri |
| `preprocessing.py` | Preparazione dei segmenti per la CNN |
| `predict_character.py` | Predizione di un singolo carattere |
| `predict_word.py` | Predizione e analisi di una singola parola |
| `spell_correct.py` | Generazione candidati e correzione ortografica |
| `language_detection.py` | Confronto tra italiano e inglese |
| `ngram_language_model.py` | Caricamento e interrogazione dei bigrammi |
| `context_scoring.py` | Correzione prudente basata sul contesto |
| `text_recognizer.py` | Coordinamento della pipeline OCR completa |
| `qt_app.py` | Interfaccia grafica |
| `build_bigram_dictionary.py` | Generazione dei dizionari di bigrammi |
| `check_model.py` | Controllo di compatibilità tra dataset e modello |
| `inspect_dataset.py` | Visualizzazione di campioni EMNIST |

## Tecnologie utilizzate

- Python
- PyTorch e Torchvision
- OpenCV
- Pillow
- NumPy
- SymSpellPy
- PySide6
- Matplotlib

## Limiti attuali

- La CNN riconosce solamente le 26 lettere dell'alfabeto latino.
- Maiuscole e minuscole vengono ricondotte alla stessa classe.
- Numeri, punteggiatura e simboli non vengono classificati.
- La qualità finale dipende dalla corretta separazione dei caratteri.
- Testi molto inclinati, caratteri sovrapposti o grafie molto connesse possono produrre segmentazioni errate.
- La correzione linguistica dipende dalla copertura dei dizionari e dei bigrammi.
- Il fine-tuning richiede trascrizioni corrette fornite manualmente.
- Il sistema non apprende automaticamente dalle immagini caricate.
