# CDC-Wastewater-Analysis-ML

CDC Atıksu verilerini kullanarak Gradient Boosting ve Zaman Serisi Özellik Mühendisliği ile Influenza A tahmini

Veri setimiz (dataset) dosyasının içinde bulunmaktadır (Veriler CDC'nin halka açık veri tabanından alınmıştır.)

Bu proje, CDC (Hastalık Kontrol ve Önleme Merkezleri) tarafından sağlanan atıksu verilerini analiz ederek İnfluenza A (Grip) virüsünün yayılım trendlerini tespit etmeyi ve tahminlemeyi amaçlar.

Gerekli Kütüphaneler : pandas, numpy, matplotlib, seaborn, scikit-learn
!!file_name değişkenini kendi dosya yolunuza göre yazmayı unutmayın!!


Gelişmiş Özellik Mühendisliği (Feature Engineering V2):

Veri Temizliği & İmputasyon: Eksik akış hızı ve nüfus verileri istatistiksel yöntemlerle (medyan/mod) dolduruldu.

Gecikme (Lag) Özellikleri: Geçmiş verilerin bugünü nasıl etkilediğini görmek için lag_1 ve lag_2 değişkenleri oluşturuldu.

Hedef Kodlama (Target Encoding): wwtp_jurisdiction (bölge) verisi, tarihsel vaka oranlarına göre sayısal hale getirildi.

Etkileşim (Interaction) Terimleri: Nüfus ve Akış Hızı gibi değişkenler çarpılarak yeni türetilmiş özellikler (pop_x_flow) eklendi.

Boyut İndirgeme (PCA): Veri setindeki varyansın %95'ini koruyarak özellik sayısı azaltıldı ve model karmaşıklığı düşürüldü.

Model Karşılaştırması: 4 farklı senaryo (PCA'lı/PCA'sız ve GB/Lineer) Accuracy, ROC-AUC ve Average Precision metrikleri ile kıyaslandı.
