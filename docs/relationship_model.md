# İlişki Modeli: Zenginleştirilmiş İlişkiler

## Genel Bakış

İlişki modelleme sistemi, knowledge graph içindeki ilişkiler için zenginleştirilmiş bir temsil sağlar. Kapsamlı bir ilişki tipi taksonomisi, sınıflandırma mekanizmaları ve yapılandırılmış ilişki detaylarını içerir.

## İlişki Yapısı

### Zenginleştirilmiş İlişki Formatı

Her ilişki şu yapıyı takip eder:
- **source**: Kaynak entity tanımlayıcısı
- **target**: Hedef entity tanımlayıcısı
- **relationship_type**: Taksonomiden tip
- **relationship_detail**: Kısa doğal dil açıklaması
- **confidence**: Güven skoru (0.0 ile 1.0 arası)
- **source_ref**: Kaynak referansı (örn. chunk_id, document_id)

**Not**: Dokümantasyon dosyalarında kod/JSON örneği tutulmaz; bu formatı uygulamada `enriched` export olarak görürsünüz.

### İlişki Detayı

`relationship_detail` alanı, ilişkinin kısa, doğal dil açıklamasını içerir. Bu şunları sağlar:
- **Açıklanabilirlik**: Bu ilişkinin neden var olduğu
- **Bağlam**: İlişkinin belirli domain'de ne anlama geldiği
- **İzlenebilirlik**: İnsan tarafından okunabilir gerekçe

**Örnek:**
```
"Alice, Bob'un karar verme sürecini stratejik tavsiyeleriyle etkiler."
```

## İlişki Tipi Taksonomisi

### Nedensel İlişkiler (Causal Relationships)

- **causes**: A doğrudan B'ye neden olur
- **caused_by**: A, B tarafından neden olunur
- **triggers**: A, B'yi tetikler
- **results_in**: A, B ile sonuçlanır

**Örnek:**
```
"Yağmur, sel'e neden oldu."
→ Yağmur causes Sel
```

### Mekansal İlişkiler (Spatial Relationships)

- **located_at**: A, B'de konumlanmıştır
- **contains**: A, B'yi içerir
- **near**: A, B'ye yakındır
- **inside/outside**: Mekansal kapsama

**Örnek:**
```
"Paris, Fransa'da konumlanmıştır."
→ Paris located_at Fransa
```

### Zaman İlişkileri (Temporal Relationships)

- **occurs_before**: A, B'den önce gerçekleşir
- **occurs_after**: A, B'den sonra gerçekleşir
- **occurs_during**: A, B sırasında gerçekleşir
- **follows**: A, B'yi takip eder

**Örnek:**
```
"Kahvaltı, öğle yemeğinden önce gelir."
→ Kahvaltı occurs_before Öğle Yemeği
```

### Sosyal/Etkileşim İlişkileri (Social/Interaction Relationships)

- **influences**: A, B'yi etkiler
- **influenced_by**: A, B'den etkilenir
- **collaborates_with**: A, B ile işbirliği yapar
- **works_with**: A, B ile çalışır
- **owns**: A, B'ye sahiptir
- **belongs_to**: A, B'ye aittir
- **member_of**: A, B'nin üyesidir

**Örnek:**
```
"Alice, Bob ile işbirliği yapar."
→ Alice collaborates_with Bob
```

### Hiyerarşik İlişkiler (Hierarchical Relationships)

- **parent_of**: A, B'nin ebeveynidir
- **child_of**: A, B'nin çocuğudur
- **part_of**: A, B'nin parçasıdır
- **has_part**: A, B parçasına sahiptir

**Örnek:**
```
"Motor, arabanın parçasıdır."
→ Motor part_of Araba
```

### Aksiyon İlişkileri (Action Relationships)

- **creates**: A, B'yi oluşturur
- **destroys**: A, B'yi yok eder
- **modifies**: A, B'yi değiştirir
- **produces**: A, B'yi üretir
- **consumes**: A, B'yi tüketir

**Örnek:**
```
"Apple, iPhone üretir."
→ Apple produces iPhone
```

### İletişim İlişkileri (Communication Relationships)

- **communicates_with**: A, B ile iletişim kurar
- **informs**: A, B'yi bilgilendirir
- **requests**: A, B'den talep eder
- **responds_to**: A, B'ye yanıt verir

**Örnek:**
```
"Alice, Bob'u bilgilendirir."
→ Alice informs Bob
```

### Duygusal/Öznel İlişkiler (Emotional/Subjective Relationships)

- **likes/dislikes**: A, B'yi sever/sevmez
- **loves/hates**: Güçlü duygusal bağlantı
- **fears**: A, B'den korkar
- **trusts**: A, B'ye güvenir

**Örnek:**
```
"Alice, Bob'u sever."
→ Alice loves Bob
```

Ve daha birçok domain-agnostic tip.

## Sınıflandırma Sistemi

### Hibrit Sınıflandırıcı Tasarımı

İlişki sınıflandırma sistemi hibrit bir yaklaşım kullanır:

#### Kural Tabanlı Sınıflandırma

- Metin üzerinde pattern eşleştirme
- Tip uyumluluğu sezgileri
- Schema kısıt kontrolü
- Hızlı ve deterministik

**Avantajlar:**
- Hızlı işlem
- Deterministik sonuçlar
- Düşük maliyet

**Sınırlamalar:**
- Karmaşık durumlar için sınırlı doğruluk
- Pattern'ler manuel olarak tanımlanmalı

#### LLM Tabanlı Sınıflandırma (Gelecek)

- Karmaşık durumlar için LLM kullanımı
- Açıklama üretimi
- Belirsiz durumlar için daha yüksek doğruluk

**Avantajlar:**
- Yüksek doğruluk
- Karmaşık pattern'leri yakalama
- Otomatik açıklama üretimi

**Sınırlamalar:**
- Yavaş (API çağrıları)
- Yüksek maliyet
- Henüz implement edilmedi

### Sınıflandırma Süreci

1. **Pattern Eşleştirme**: Ham metni pattern'lere karşı eşleştir
2. **Tip Sezgileri**: Entity tiplerini kullanarak olası ilişkileri çıkar
3. **Schema Kısıtları**: Schema ile tanımlı geçerli ilişkileri filtrele
4. **LLM Sınıflandırma**: (Gelecek) Zor durumlar için LLM kullan
5. **Güven Skorlama**: Kullanılan yönteme göre güven skoru ata

**Örnek Süreç:**
```
Metin: "Alice, Bob'a tavsiye verdi."
1. Pattern eşleştirme: "tavsiye verdi" → informs pattern'i bulundu
2. Tip sezgileri: Person → Person ilişkisi geçerli
3. Schema kısıtları: informs ilişkisi schema'da tanımlı
4. Sonuç: Alice informs Bob (confidence: 0.9)
```

## Dataset Bağımsızlığı

İlişki modeli, domain'ler arası çalışacak şekilde tasarlanmıştır:

- **Taksonomi**: Genel amaçlı ilişki tipleri
- **Domain Varsayımı Yok**: Hikaye, gerçekçi, teknik metinlerle çalışır
- **Genişletilebilir**: Yeni ilişki tipleri eklenebilir
- **Esnek Detaylar**: `relationship_detail` domain'e uyum sağlar

## Kullanım Senaryoları

### Hikaye Metinleri

- **Karakter ilişkileri**: influences, collaborates_with
- **Hikaye ilişkileri**: caused_by, triggers
- **Mekansal ilişkiler**: located_at, visits

**Örnek:**
```
"Alice, Bob'u etkiledi ve birlikte çalıştılar."
→ Alice influences Bob
→ Alice collaborates_with Bob
```

### Gerçekçi Metinler

- **Tarihi ilişkiler**: influences, member_of
- **Nedensel ilişkiler**: causes, results_in
- **Zaman ilişkileri**: occurs_before, occurs_during

**Örnek:**
```
"Einstein, 1905'te özel görelilik teorisini geliştirdi."
→ Einstein occurs_during 1905
→ Einstein creates Özel Görelilik Teorisi
```

### Teknik Dokümanlar

- **Sistem ilişkileri**: contains, part_of
- **Süreç ilişkileri**: produces, consumes
- **Bağımlılık ilişkileri**: depends_on, uses

**Örnek:**
```
"Sistem, veritabanından veri çeker ve API'ye gönderir."
→ Sistem consumes Veritabanı
→ Sistem produces API
```

## Trade-off'lar

### Avantajlar

- **Zengin Taksonomi**: Birçok ilişki tipini kapsar
- **Açıklanabilir İlişkiler**: Detay alanı aracılığıyla açıklanabilir ilişkiler
- **Hibrit Sınıflandırma**: Hız ve doğruluk dengesi
- **Dataset-Agnostic Tasarım**: Farklı domain'lerle uyumlu

### Sınırlamalar

- **Domain-Specific İlişkiler**: Taksonomi tüm domain-specific ilişkileri kapsamayabilir
- **Kural Tabanlı Sınırlamalar**: Kural tabanlı sınıflandırma sınırlı doğruluğa sahip
- **LLM Implementasyonu**: LLM tabanlı sınıflandırma henüz implement edilmedi
- **Detay Üretimi**: `relationship_detail` üretimi harici mantık gerektirir

## Gelecek Geliştirmeler

Potansiyel iyileştirmeler:

1. **LLM Tabanlı Sınıflandırma**: Karmaşık durumlar için LLM implementasyonu
2. **Otomatik Detay Üretimi**: `relationship_detail` için otomatik üretim
3. **Domain-Specific Taksonomi**: Domain-specific taksonomi uzantıları
4. **Güven Kalibrasyonu**: İlişki güven skorlarının kalibrasyonu
5. **Validasyon Kuralları**: İlişki validasyon kuralları
6. **Çoklu Dil Desteği**: Çoklu dil ilişki tanımlamaları
7. **İlişki Öğrenme**: Metinden otomatik ilişki öğrenme

## Best Practices

### İlişki Tanımlama

1. **Açıklayıcı Detaylar**: `relationship_detail` alanı açıklayıcı olmalı
2. **Uygun Tip Seçimi**: İlişki için en uygun tipi seçin
3. **Güven Skorlama**: Güven skorlarını doğru atayın
4. **Kaynak Referansı**: Her ilişki için kaynak referansı sağlayın

### Sınıflandırma

1. **Pattern Kütüphanesi**: Kapsamlı bir pattern kütüphanesi oluşturun
2. **Tip Uyumluluğu**: Entity tiplerini kullanarak ilişki geçerliliğini kontrol edin
3. **Schema Kısıtları**: Schema kısıtlarını kullanarak geçersiz ilişkileri filtreleyin
4. **Güven Eşikleri**: Güven eşiklerini domain'e göre ayarlayın

### İlişki Kalitesi

1. **Doğruluk Kontrolü**: İlişkilerin doğruluğunu kontrol edin
2. **Tutarlılık**: Benzer ilişkiler için tutarlı tip kullanın
3. **Açıklanabilirlik**: İlişkilerin açıklanabilir olmasını sağlayın
4. **İzlenebilirlik**: Her ilişki için kaynak referansı sağlayın
