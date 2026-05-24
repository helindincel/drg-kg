# Schema Tasarımı: Dataset-Agnostic Entity Schema Oluşturma

## Genel Bakış

Schema oluşturma sistemi, yapılandırılmış özelliklerle entity sınıfları tanımlamak için dataset-agnostic bir yaklaşım sunar. Bu sistem, core schema yapısında domain-specific değişiklikler gerektirmeden farklı domain'ler ve metin türlerinde çalışacak şekilde tasarlanmıştır.

## Tasarım Prensipleri

### Dataset-Agnostic Mimari

Schema oluşturma sistemi şu prensipleri takip eder:

1. **Genelleştirme Önceliği**: Entity sınıfları, domain-specific detaylar yerine domain'ler arası ortak pattern'leri yakalayacak şekilde tasarlanmıştır.

2. **Özellik Tabanlı Esneklik**: Domain-specific attribute'ları hard-code etmek yerine, sistem her özelliğin şunlara sahip olduğu esnek bir özellik sistemi kullanır:
   - `name`: Özellik için benzersiz tanımlayıcı
   - `description`: İnsan tarafından okunabilir açıklama
   - `example_value`: Beklenen veri tipi ve formatı gösteren somut örnek

3. **Genişletilebilirlik**: Varsayılan entity sınıfları (Person, Location, Event) sağlanırken, sistem core kodu değiştirmeden özel entity sınıflarının kolayca eklenmesine izin verir.

## Entity Sınıf Yapısı

### Varsayılan Entity Sınıfları

#### Person (Kişi)

İnsan bireyleri psikolojik ve ilişkisel özelliklerle temsil eder:
- **emotion**: Mevcut duygusal durum
- **intent**: Ana hedefler veya niyetler
- **traits**: Karakter özellikleri (liste)
- **relationships**: Diğerleriyle ilişkiler (sözlük)
- **role**: Bağlamdaki pozisyon veya rol
- **age**: Yaş veya yaş aralığı

**Kullanım Örnekleri:**
- Hikaye karakterleri
- Tarihi figürler
- Biyografik metinlerdeki kişiler

#### Location (Konum)

Mekanları mekansal ve sembolik anlamla temsil eder:
- **atmosphere**: Duygusal/duyusal atmosfer
- **symbolism**: Sembolik temsil
- **type**: Konum kategorisi
- **coordinates**: Coğrafi koordinatlar (opsiyonel)
- **features**: Dikkat çekici özellikler (liste)

**Kullanım Örnekleri:**
- Hikaye mekanları
- Coğrafi konumlar
- Tarihi yerler

#### Event (Olay)

Temporal ve nedensel yönleri olan olayları temsil eder:
- **actors**: İlgili entity'ler (liste)
- **outcomes**: Sonuçlar veya etkiler
- **temporal_scope**: Zaman periyodu veya süresi
- **type**: Olay kategorisi
- **significance**: Önem seviyesi
- **cause**: Tetikleyici veya neden

**Kullanım Örnekleri:**
- Hikaye olayları
- Tarihi olaylar
- Teknik dokümanlardaki süreçler

## Özellik Tanımlama Sistemi

Her entity sınıfındaki özellik yapılandırılmış bir formatta tanımlanır:

```python
PropertyDefinition(
    name: str,              # Özellik adı
    description: str,       # Açıklama
    example_value: Any      # Örnek değer
)
```

`example_value` herhangi bir Python tipi (str, int, list, dict, vb.) olabilir, bu da veri temsilinde esneklik sağlar.

**Örnek Özellik Tanımı:**

```python
PropertyDefinition(
    name="emotion",
    description="Kişinin mevcut duygusal durumu",
    example_value="mutlu"
)
```

## Schema Export Formatları

### JSON Formatı

Standart JSON temsili, şunlar için uygundur:
- Konfigürasyon depolama
- API yanıtları
- Sistemler arası iletişim

**Örnek:**
```json
{
  "entity_classes": [
    {
      "name": "Person",
      "properties": [
        {
          "name": "emotion",
          "description": "Mevcut duygusal durum",
          "example_value": "mutlu"
        }
      ]
    }
  ]
}
```

### YAML Formatı

İnsan tarafından okunabilir YAML formatı, şunlar için uygundur:
- Manuel düzenleme
- Dokümantasyon
- Konfigürasyon dosyaları

**Örnek:**
```yaml
entity_classes:
  - name: Person
    properties:
      - name: emotion
        description: Mevcut duygusal durum
        example_value: mutlu
```

## Genişletme Mekanizması

### Özel Entity Sınıfları Ekleme

Yeni entity sınıfları `add_entity_class()` metodu ile eklenebilir:

1. `EntityClassDefinition` instance'ı oluştur
2. `add_property()` kullanarak özellikler ekle
3. Schema generator'a kaydet

**Örnek:**
```python
# Yeni bir "Company" entity sınıfı ekle
company_class = EntityClassDefinition(name="Company")
company_class.add_property("name", "Şirket adı", "Apple Inc.")
company_class.add_property("industry", "Sektör", "Teknoloji")
schema.add_entity_class(company_class)
```

### Mevcut Sınıfları Değiştirme

Mevcut entity sınıfları şu yöntemlerle güncellenebilir:
- `update_entity_class()`: Tüm sınıf tanımını değiştir
- Doğrudan özellik manipülasyonu (validasyon ile)

## Kullanım Senaryoları

### Hikaye Metinleri

- **Person entity'leri**: Karakter gelişimini yakalar
- **Location entity'leri**: Mekan ve atmosferi yakalar
- **Event entity'leri**: Hikaye noktalarını yakalar

**Örnek:**
```
"Alice, mutlu bir ruh haliyle, Paris'teki kafede oturuyordu. 
Aniden bir patlama sesi duyuldu."
```
- Person: Alice (emotion: mutlu)
- Location: Paris (type: şehir)
- Event: Patlama (type: olay, significance: yüksek)

### Gerçekçi Metinler

- **Person entity'leri**: Tarihi figürleri temsil eder
- **Location entity'leri**: Coğrafi entity'leri temsil eder
- **Event entity'leri**: Tarihi olayları temsil eder

**Örnek:**
```
"Einstein, 1905'te İsviçre'de özel görelilik teorisini geliştirdi."
```
- Person: Einstein (role: fizikçi)
- Location: İsviçre (type: ülke)
- Event: Teori geliştirme (temporal_scope: 1905)

### Teknik Dokümanlar

- Teknik entity sınıfları ile genişletilebilir
- Özellikler teknik kavramlara uyarlanır

**Örnek:**
```
"Sistem, veritabanından veri çeker ve API'ye gönderir."
```
- Teknik entity sınıfları: System, Database, API
- İlişkiler: çeker, gönderir

## Trade-off'lar

### Avantajlar

- **Çoklu Domain Desteği**: Modifikasyon olmadan birden fazla domain'de çalışır
- **Esnek Özellik Sistemi**: Çeşitli veri tiplerini barındırır
- **Kolay Genişletilebilirlik**: Yeni entity sınıfları kolayca eklenebilir
- **Dataset-Agnostic**: Farklı veri kaynaklarıyla uyumlu

### Sınırlamalar

- **Domain-Specific Kavramlar**: Domain-specific kavramlar için özel özellikler gerekebilir
- **Statik Örnekler**: Örnek değerler statiktir ve tüm olası varyasyonları yakalamaz
- **Temel Validasyon**: Özellik validasyonu temeldir (runtime'da tip kontrolü)
- **Manuel Genişletme**: Yeni entity sınıfları manuel olarak eklenmelidir

## Gelecek Geliştirmeler

Potansiyel iyileştirmeler:

1. **Özellik Tip Validasyonu**: Schema seviyesinde tip kısıtlamaları
2. **Özellik İlişkileri**: Özellikler arası bağımlılıklar
3. **Dinamik Örnek Üretimi**: Bağlama dayalı dinamik örnek üretimi
4. **Çoklu Dil Desteği**: Çoklu dil özellik tanımlamaları
5. **Otomatik Schema Çıkarımı**: Metinden otomatik schema çıkarımı
6. **Schema Versiyonlama**: Schema değişikliklerini takip etme

## Best Practices

### Entity Sınıfı Tasarımı

1. **Genel Özellikler**: Domain-specific detaylar yerine genel pattern'leri yakalayın
2. **Açıklayıcı İsimler**: Özellik isimleri açıklayıcı olmalı
3. **Anlamlı Örnekler**: Örnek değerler gerçek kullanım senaryolarını yansıtmalı
4. **Tutarlılık**: Benzer entity sınıfları için tutarlı özellik isimleri kullanın

### Özellik Tanımlama

1. **Açıklayıcı Açıklamalar**: Her özellik için net açıklamalar sağlayın
2. **Uygun Veri Tipleri**: Özellik için uygun veri tipini seçin
3. **Örnek Değerler**: Gerçekçi ve anlamlı örnek değerler kullanın

### Schema Yönetimi

1. **Versiyonlama**: Schema değişikliklerini versiyonlayın
2. **Dokümantasyon**: Schema değişikliklerini dokümante edin
3. **Test Etme**: Yeni schema'ları test edin
4. **Geriye Dönük Uyumluluk**: Mümkün olduğunca geriye dönük uyumluluğu koruyun
