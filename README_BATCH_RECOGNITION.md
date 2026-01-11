# Batch Face Recognition System

Ovaj dokument opisuje novi batch face recognition sistem koji organizuje slike u batch-eve od po 5000 slika za optimizovanu paralelnu pretragu.

## 🎯 **PREGLED SISTEMA**

Novi sistem je **potpuno nezavistan** od postojeće logike i dodaje sledeće komponente:

### **Krerani fajlovi:**
- `app/services/batch_management_service.py` - Upravljanje batch strukturom
- `app/services/batch_recognition_service.py` - Paralelno prepoznavanje kroz batch-eve
- `app/controllers/batch_recognition_controller.py` - API controller za batch endpointe
- `app/routes/batch_recognition_routes.py` - Flask rute za batch API
- `scripts/batch_migration_command.py` - CLI komanda za migraciju postojećih slika
- `README_BATCH_RECOGNITION.md` - Ova dokumentacija

### **Izmenjeni fajlovi:**
- `app/__init__.py` - Registrovan novi batch_recognition_bp blueprint

## 📁 **STRUKTURA BATCH FOLDERA**

```
storage/recognized_faces_batched/{domain}/
├── batch_0001/          # Slike 1-5000 + representations_vgg_face.pkl
├── batch_0002/          # Slike 5001-10000 + representations_vgg_face.pkl  
├── batch_0003/          # Slike 10001-15000 + representations_vgg_face.pkl
├── batch_metadata.json  # Informacije o batch-evima
└── ...
```

## 🚀 **KORIŠĆENJE**

### **1. Migracija postojećih slika u batch strukturu**

```bash
# Dry run (samo provera) za jedan domain
python scripts/batch_migration_command.py --domain example.com --dry-run

# Osnovna migracija (kopira slike, kreira pickle fajlove)
python scripts/batch_migration_command.py --domain example.com

# Migracija sa brisanjem originalnih slika
python scripts/batch_migration_command.py --domain example.com --delete-originals

# Brža migracija bez pickle fajlova (batch recognition neće raditi!)
python scripts/batch_migration_command.py --domain example.com --no-pickle

# Kompletnа migracija sa brisanjem originalnih i pickle fajlovima
python scripts/batch_migration_command.py --domain example.com --delete-originals

# Migracija za sve domain-e
python scripts/batch_migration_command.py --all-domains --delete-originals

# Pregled informacija o postojećim batch-evima
python scripts/batch_migration_command.py --info --domain example.com

# Lista svih domain-a sa batch strukturom
python scripts/batch_migration_command.py --list-batch-domains
```

#### **⚠️ VAŽNE OPCIJE:**

- **`--delete-originals`**: Briše originalne slike nakon kopiranja u batch-eve
- **`--no-pickle`**: Ne kreira pickle fajlove (brže, ali batch recognition neće raditi)
- **`--force`**: Prepisuje postojeće batch-eve
- **`--dry-run`**: Samo pokazuje šta bi se uradilo, bez stvarnih izmena

### **2. Batch Face Recognition API**

#### **Glavni endpoint za prepoznavanje:**
```bash
# POST /api/batch/recognize
curl -X POST \
     -F "file=@test_image.jpg" \
     -F "domain=example.com" \
     -F "max_threads=3" \
     http://localhost:5000/api/batch/recognize
```

#### **Statistike o batch strukturi:**
```bash
# GET /api/batch/stats?domain=example.com
curl "http://localhost:5000/api/batch/stats?domain=example.com"
```

#### **Lista svih domain-a sa batch strukturom:**
```bash
# GET /api/batch/domains
curl "http://localhost:5000/api/batch/domains"
```

#### **Detaljne informacije o batch strukturi:**
```bash
# GET /api/batch/info?domain=example.com
curl "http://localhost:5000/api/batch/info?domain=example.com"
```

#### **Health check:**
```bash
# GET /api/batch/health
curl "http://localhost:5000/api/batch/health"
```

## ⚡ **PERFORMANCE PREDNOSTI**

1. **Paralelizacija**: Do 3 batch-a se procesiraju simultano
2. **Skalabilnost**: Lako dodavanje novih batch-eva
3. **Optimizacija**: Svaki batch ima svoj pickle fajl
4. **Existing compatibility**: Postojeća logika ostaje netaknuta

## 🔧 **TEHNIČKI DETALJI**

### **Korišćene postojeće komponente:**
- `RecognitionService.clean_domain_for_path()` - Čišćenje domain-a
- `RecognitionService.process_single_face()` - Validacija lica
- `RecognitionService.analyze_recognition_results()` - Analiza rezultata
- `ImageService.resize_image()` - Smanjivanje slika
- `FaceValidationService.process_face_filtering()` - Filtriranje lica
- `DeepFace.find()` - Face recognition algoritam

### **Thread konfiguracija:**
- Maksimalno 3 batch-a simultano (konfigurabilno)
- 5 minuta timeout po batch-u
- ThreadPoolExecutor za paralelizaciju

### **Batch organizacija:**
- 5000 slika po batch-u (konfigurabilno)
- Automatsko kreiranje metadata fajla
- Konzistentna struktura foldera

## 📊 **PRIMER RESPONSE-A**

### **Uspešno prepoznavanje:**
```json
{
  "status": "success",
  "message": "Face recognized as: Marko Petrovic",
  "person": "Marko Petrovic",
  "recognized_persons": [
    {
      "name": "Marko Petrovic",
      "face_coordinates": {
        "x_percent": 45.2,
        "y_percent": 23.1,
        "width_percent": 15.3,
        "height_percent": 20.7
      }
    }
  ],
  "batch_processing": {
    "total_processing_time": 4.52,
    "batch_summary": {
      "total_batches": 3,
      "processed_batches": 3,
      "failed_batches": 0,
      "total_images_searched": 12450
    }
  },
  "api_info": {
    "api_version": "batch_v1",
    "request_processing_time": 4.53
  }
}
```

## 🛠️ **ODRŽAVANJE**

### **Dodavanje novih slika:**
Kada se dodaju nove slike u postojeći domain, treba:
1. Pokrenuti batch migraciju ponovo sa `--force` flag-om
2. Ili manuelno kopirati nove slike u odgovarajući batch

### **Monitoring:**
- Koristiti `/api/batch/health` za monitoring sistema
- Koristiti `/api/batch/stats` za praćenje performance-a

## ⚠️ **NAPOMENE**

1. **Postojeća logika ostaje netaknuta** - sve radi paralelno
2. **Batch sistem je opcioni** - može se koristiti kada je potreban
3. **CLI komanda podržava dry-run** za bezbednu proveru
4. **Thread broj je ograničen** da ne preoptereti sistem
5. **Metadata se automatski ažurira** tokom migracije

## 🗂️ **UPRAVLJANJE ORIGINALNIM SLIKAMA**

### **Default ponašanje:**
- **Originalne slike se NE brišu** - samo se kopiraju u batch strukturu
- **Pickle fajlovi se automatski kreiraju** za svaki batch

### **Opcije za brisanje:**
```bash
# Briši originalne slike nakon kopiranja
python scripts/batch_migration_command.py --domain example.com --delete-originals
```

### **Opcije za pickle fajlove:**
```bash
# Preskačemo kreiranje pickle fajlova (brže, ali batch recognition neće raditi)
python scripts/batch_migration_command.py --domain example.com --no-pickle
```

## 🔧 **PICKLE FAJLOVI**

Svaki batch folder **mora** da ima `representations_vgg_face.pkl` fajl da bi batch recognition radio.

### **Automatska kreacija:**
- Sistem automatski poziva `DeepFace.find()` sa test slikom iz batch-a
- Kreira se pickle fajl za optimizovanu pretragu
- Ovo može potrajati nekoliko minuta po batch-u

### **Manuelna kreacija:**
Ako je batch kreiran sa `--no-pickle`, možete naknadno kreirati pickle fajlove:
```bash
# Pokreni ponovo sa pickle kreacijom
python scripts/batch_migration_command.py --domain example.com --force
```

## 🔍 **TROUBLESHOOTING**

### **Ako batch recognition ne radi:**
1. Proveriti da li je batch struktura kreirana: `--list-batch-domains`
2. Pokrenuti health check: `/api/batch/health`
3. Validirati batch metadata: `--info --domain your.domain`

### **Performance problemi:**
1. Smanjiti `max_threads` parametar
2. Proveriti veličinu batch-eva
3. Monitorovati memory usage

---

🎉 **Sistem je spreman za korišćenje!** 