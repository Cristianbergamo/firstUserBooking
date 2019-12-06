La cartella contiene:
 - submission.csv - file contenente le previsioni del modello
 - main.py - file python contenente il codice necessario a ottenere il file submission.csv più tutto il codice utilizzato nella fase di training
 - First User Booking - ConTe.pdf - presentazione dell'approccio al problema e dei risultati ottenuti per un pubblico non tecnico
 - ExploratoryAnalysis (jupyter notebook).ipynb - notebook contenente l'exploratory analysis richiesta con, in fondo, spiegazione del processo di trattamento dati realizzato
 - validationLogs - cartella contenente i log del processo di cross validation, dai quali si evincono gli iperparametri migliori
 - models - cartella contenente tutti i file binari necessari ad ottenere il file submission.csv, compreso il file relativo al modello addestrato
 - data - cartella vuota - inserire qui il file di test se si vuole replicare il file submission.csv

Per replicare e verificare la veridicità del file submission, è sufficiente inserire il file di test, così come fornito per la challenge, nella cartella data ed eseguire il file main.py.

