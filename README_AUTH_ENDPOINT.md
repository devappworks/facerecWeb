# Email-to-Token Authentication Endpoint

Ovaj dokument opisuje API endpoint koji omogućava dobijanje autentifikacionog tokena na osnovu email adrese korisnika.

## Pregled

Endpoint prima email adresu sa fronta, pronalazi odgovarajući key(s) u `CLIENTS_EMAILS` JSON-u, a zatim koristi taj/te key(s) da pronađe token(s) u postojećem `CLIENTS_TOKENS` JSON-u.

**Nova funkcionalnost:** Endpoint sada podržava više domena po email adresi - ako imate jedan rezultat, vraća se u standardnom formatu, a ako imate dva ili više rezultata, vraća se niz tokena.

## Multi-Domain Podrška

Endpoint sada podržava konfiguraciju gde jedan email može imati tokene za različite domene:

### Jednostruka domena (postojeća funkcionalnost):
```bash
CLIENTS_EMAILS={"rts@gmail.com": "rts", "hrt@gmail.com": "hrt"}
```

### Višestruke domene (nova funkcionalnost):
```bash
CLIENTS_EMAILS={"rts@gmail.com": ["rts", "rts_domain2"], "hrt@gmail.com": "hrt"}
```

**Ponašanje:**
- **Jedan token:** Vraća se u standardnom formatu (backwards compatible)
- **Više tokena:** Vraća se kao niz objekata sa dodatnim `domain` poljem

## Vaše Environment Varijable

Vaše postojeće environment varijable:

```bash
CLIENTS_EMAILS={"rts@gmail.com": "rts", "hrt@gmail.com": "hrt", "test@gmail.com": "test", "nikola1jankovic@gmail.com": "media24"}

CLIENTS_TOKENS={"dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD": "rts", "K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh": "hrt", "EiasPl9oJWe7Ps6j5AW94DA5IXqaGCh2Seg": "test", "d9OLEFYdx18bUTGkIpaKyDFCcko1jYu0Ha1": "media24"}
```

## Token Mapiranje

Na osnovu vaših environment varijabli, očekivani token mapping je:

| Email | Key | Token |
|-------|-----|-------|
| rts@gmail.com | rts | dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD |
| hrt@gmail.com | hrt | K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh |
| test@gmail.com | test | EiasPl9oJWe7Ps6j5AW94DA5IXqaGCh2Seg |
| nikola1jankovic@gmail.com | media24 | d9OLEFYdx18bUTGkIpaKyDFCcko1jYu0Ha1 |

## API Endpoints

### 1. Dobijanje Tokena po Email-u

**Endpoint:** `POST /api/auth/token-by-email`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
    "email": "rts@gmail.com"
}
```

**Uspešan Odgovor (200 OK):**

**Jednostruka domena (postojeći format):**
```json
{
    "success": true,
    "data": {
        "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
        "email": "rts@gmail.com"
    }
}
```

**Višestruke domene (novi format):**
```json
{
    "success": true,
    "data": [
        {
            "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
            "email": "rts@gmail.com",
            "domain": "rts"
        },
        {
            "token": "anotherTokenForDifferentDomain123456789",
            "email": "rts@gmail.com",
            "domain": "rts_domain2"
        }
    ]
}
```

**Greške:**

**404 Not Found - Email nije pronađen:**
```json
{
    "success": false,
    "error": "Email 'unknown@gmail.com' not found in authorized users"
}
```

**500 Internal Server Error - Token mapiranje problema:**
```json
{
    "success": false,
    "error": "Token not found for key 'rts'. Please contact administrator"
}
```

**400 Bad Request - Neispravan zahtev:**
```json
{
    "success": false,
    "error": "Email field is required"
}
```

### 2. Validacija Email Pristupa

**Endpoint:** `POST /api/auth/validate-email`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
    "email": "rts@gmail.com"
}
```

**Odgovor (200 OK):**
```json
{
    "success": true,
    "data": {
        "email": "rts@gmail.com",
        "has_access": true,
        "key": "rts"
    }
}
```

### 3. Health Check

**Endpoint:** `GET /api/auth/health`

**Odgovor (200 OK):**
```json
{
    "success": true,
    "message": "Authentication service is running",
    "service_available": true
}
```

## Kako funkcioniše

1. **Frontend šalje email** na `/api/auth/token-by-email`
2. **Service proverava CLIENTS_EMAILS** da pronađe odgovarajući key(s) za email
3. **Service traži token(s)** u CLIENTS_TOKENS koristeći key(s) iz prethodnog koraka
4. **Vraća token(s)** u odgovarajućem formatu ili grešku

### Primer logike - jednostruka domena:

```
Email: "rts@gmail.com"
↓
CLIENTS_EMAILS lookup: "rts@gmail.com" → "rts" 
↓
CLIENTS_TOKENS reverse lookup: "rts" → "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD"
↓
Return single format: {"token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD", "email": "rts@gmail.com"}
```

### Primer logike - višestruke domene:

```
Email: "rts@gmail.com"
↓
CLIENTS_EMAILS lookup: "rts@gmail.com" → ["rts", "rts_domain2"]
↓
CLIENTS_TOKENS reverse lookup: 
  - "rts" → "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD"
  - "rts_domain2" → "anotherTokenForDifferentDomain123456789"
↓
Return array format: [
  {"token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD", "email": "rts@gmail.com", "domain": "rts"},
  {"token": "anotherTokenForDifferentDomain123456789", "email": "rts@gmail.com", "domain": "rts_domain2"}
]
```

## Testiranje

### Pokretanje aplikacije
```bash
python run.py
```

### Pokretanje test script-a
```bash
python test_auth_endpoint.py
```

### Manual testiranje sa curl

**Test za rts@gmail.com:**
```bash
curl -X POST http://localhost:5000/api/auth/token-by-email \
  -H "Content-Type: application/json" \
  -d '{"email": "rts@gmail.com"}'
```

**Očekivani odgovor:**
```json
{
  "success": true,
  "data": {
    "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
    "email": "rts@gmail.com"
  }
}
```

**Test za nepostojeci email:**
```bash
curl -X POST http://localhost:5000/api/auth/token-by-email \
  -H "Content-Type: application/json" \
  -d '{"email": "nepostoji@gmail.com"}'
```

**Očekivani odgovor:**
```json
{
  "success": false,
  "error": "Email 'nepostoji@gmail.com' not found in authorized users"
}
```

## Frontend Integracija

### JavaScript/Fetch primer

```javascript
async function getTokenByEmail(email) {
    try {
        const response = await fetch('/api/auth/token-by-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: email })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Handle both single token and multiple tokens response
            if (Array.isArray(data.data)) {
                // Multiple domains - array response
                console.log('Multiple tokens retrieved:', data.data);
                return data.data; // Return array of token objects
            } else {
                // Single domain - object response  
                console.log('Single token retrieved:', data.data.token);
                return data.data; // Return single token object
            }
        } else {
            console.error('Error:', data.error);
            throw new Error(data.error);
        }
    } catch (error) {
        console.error('Request failed:', error);
        throw error;
    }
}

// Primer korišćenja za jednostruku domenu
getTokenByEmail('rts@gmail.com')
    .then(result => {
        if (Array.isArray(result)) {
            // Multiple domains
            result.forEach((tokenData, index) => {
                console.log(`Token ${index + 1} for domain ${tokenData.domain}:`, tokenData.token);
                localStorage.setItem(`authToken_${tokenData.domain}`, tokenData.token);
            });
        } else {
            // Single domain
            localStorage.setItem('authToken', result.token);
            console.log('Token saved:', result.token);
        }
    })
    .catch(error => {
        // Handle greške
        alert('Neuspešno dobijanje tokena: ' + error.message);
    });
```

### React Hook primer

```javascript
import { useState, useCallback } from 'react';

export const useAuthToken = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const getTokenByEmail = useCallback(async (email) => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch('/api/auth/token-by-email', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Handle both single token and multiple tokens response
                if (Array.isArray(data.data)) {
                    // Multiple domains - array response
                    console.log('Multiple tokens retrieved:', data.data);
                    return data.data; // Return array of token objects
                } else {
                    // Single domain - object response  
                    console.log('Single token retrieved:', data.data.token);
                    return data.data; // Return single token object
                }
            } else {
                throw new Error(data.error);
            }
        } catch (err) {
            setError(err.message);
            setLoading(false);
            throw err;
        }
    }, []);
    
    return { getTokenByEmail, loading, error };
};

// Korišćenje u komponenti
function LoginForm() {
    const { getTokenByEmail, loading, error } = useAuthToken();
    const [email, setEmail] = useState('');
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        try {
            const result = await getTokenByEmail(email);
            if (Array.isArray(result)) {
                // Multiple domains
                result.forEach((tokenData, index) => {
                    console.log(`Token ${index + 1} for domain ${tokenData.domain}:`, tokenData.token);
                    localStorage.setItem(`authToken_${tokenData.domain}`, tokenData.token);
                });
            } else {
                // Single domain
                localStorage.setItem('authToken', result.token);
                console.log('Token saved:', result.token);
            }
        } catch (error) {
            console.error('Login failed:', error);
        }
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input 
                type="email" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email"
                required
            />
            <button type="submit" disabled={loading}>
                {loading ? 'Getting Token...' : 'Get Token'}
            </button>
            {error && <div className="error">{error}</div>}
        </form>
    );
}
```

## Sigurnost

1. **Email normalizacija**: Email adrese se konvertuju u mala slova i uklanjaju se space karakteri
2. **Logovanje**: Svi pokušaji pristupa se loguju za sigurnosno praćenje
3. **Error handling**: Detaljne greške se loguju, ali korisnicima se vraćaju generičke poruke
4. **Validacija**: Striktna validacija input parametara

## Test Scenariji

Test script će testirati sledeće scenarije:

### Validni Email-ovi (trebaju da vrate tokene):
- `rts@gmail.com` → `dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD`
- `hrt@gmail.com` → `K8XZ40eX1WF1v49aukU7t0hF0XO57IdZRTh`
- `test@gmail.com` → `EiasPl9oJWe7Ps6j5AW94DA5IXqaGCh2Seg`
- `nikola1jankovic@gmail.com` → `d9OLEFYdx18bUTGkIpaKyDFCcko1jYu0Ha1`

### Nevalidni Email-ovi (trebaju da vrate 404):
- `nepostoji@gmail.com`
- `admin@test.com`
- `invalid.email`
- prazan string

### Edge Cases (test case sensitivity i whitespace):
- `RTS@GMAIL.COM` → treba da radi
- `  rts@gmail.com  ` → treba da radi
- `NIKOLA1JANKOVIC@GMAIL.COM` → treba da radi

## Struktura Implementacije

```
app/
├── services/
│   └── email_token_service.py     # Logika za email-to-token mapiranje
├── controllers/
│   └── auth_controller.py         # API kontroler
├── routes/
│   └── auth_routes.py            # Flask rute
└── __init__.py                   # Registracija novih ruta
```

## Troubleshooting

### Česte Greške

1. **"Authentication service is not available"**
   - Proverite da li su environment varijable CLIENTS_EMAILS i CLIENTS_TOKENS pravilno postavljena
   - Proverite JSON format u environment varijablama

2. **"Email not found in authorized users"**
   - Proverite da li je email pravilno napisan
   - Proverite da li email postoji u CLIENTS_EMAILS

3. **"Token not found for key"**
   - Proverite da li je token mapiranje ispravno u CLIENTS_TOKENS
   - Proverite konzistentnost key-jeva između CLIENTS_EMAILS i CLIENTS_TOKENS

### Debug

Za debug informacije, možete dodati logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Aplikacija će logovati sve aktivnosti uključujući:
- Pokušaje pristupa
- Greške u mapiranju
- Uspešne token retrieval-e