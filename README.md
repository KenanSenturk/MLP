# Yapay Sinir Ağları ile BankNot Doğrulama

## Özet (Abstract)
Bu proje, banka notlarının gerçek olup olmadığını tespit etmek için çeşitli çok katmanlı algılayıcı (Multi-Layer Perceptron - MLP) modelleri geliştirmeyi ve karşılaştırmayı amaçlamaktadır. İki katmanlı MLP, üç katmanlı MLP ve scikit-learn kütüphanesinin MLPClassifier'ı kullanılarak doğrulama performansları incelenmiştir.

## 1. Giriş (Introduction)
Sahte banknotların tespiti, finansal güvenlik açısından kritik bir konudur. Bu çalışmada, Banka Notu Doğrulama veri seti üzerinde farklı MLP mimarilerinin performansını analiz ediyoruz. Kullanılan veri seti, banka notlarından alınan görüntülerin çeşitli özelliklerini içermektedir ve her bir not "gerçek" veya "sahte" olarak sınıflandırılmıştır.

### 1.1 Amaç
Bu projenin amacı:
- Sıfırdan 2 katmanlı MLP modeli oluşturmak
- Sıfırdan 3 katmanlı MLP modeli oluşturmak
- Scikit-learn kütüphanesinin MLPClassifier sınıfı ile model oluşturmak
- Bu modellerin doğruluk oranlarını karşılaştırmak

### 1.2 Veri Seti
Bu çalışmada kullanılan veri seti, banka notlarının çeşitli özelliklerini içeren "BankNote_Authentication.csv" dosyasıdır. Veri seti dört özellik ve bir sınıf etiketi içermektedir:
- Varyans (dalgacık dönüştürülmüş görüntünün varyansı)
- Çarpıklık (dalgacık dönüştürülmüş görüntünün çarpıklığı)
- Basıklık (dalgacık dönüştürülmüş görüntünün basıklığı)
- Entropi (dalgacık dönüştürülmüş görüntünün entropisi)
- Sınıf (0: gerçek, 1: sahte)

## 2. Metot (Methods)
Projede üç farklı MLP yaklaşımı uygulanmıştır:

### 2.1 İki Katmanlı MLP
İki katmanlı MLP modeli, bir gizli katman ve bir çıkış katmanından oluşur. Model şu bileşenleri içerir:
- **Giriş katmanı**: Veri setindeki özellik sayısına göre 4 nöron
- **Gizli katman**: 6 nöron, ReLU aktivasyon fonksiyonu
- **Çıkış katmanı**: 1 nöron, Sigmoid aktivasyon fonksiyonu

Modelin bileşenleri:
- Ağırlıkların He başlatması kullanılarak başlatılması
- ReLU aktivasyon fonksiyonu
- İkili çapraz entropi kayıp fonksiyonu
- 0.003 öğrenme oranı

### 2.2 Üç Katmanlı MLP
Üç katmanlı MLP modeli, iki gizli katman ve bir çıkış katmanından oluşur:
- **Giriş katmanı**: 4 nöron
- **Birinci gizli katman**: 6 nöron, ReLU aktivasyon fonksiyonu
- **İkinci gizli katman**: 6 nöron, ReLU aktivasyon fonksiyonu
- **Çıkış katmanı**: 1 nöron, Sigmoid aktivasyon fonksiyonu

Modelin bileşenleri:
- Ağırlıkların He başlatması kullanılarak başlatılması
- ReLU aktivasyon fonksiyonu
- İkili çapraz entropi kayıp fonksiyonu
- 0.003 öğrenme oranı

### 2.3 Scikit-learn MLPClassifier
Scikit-learn kütüphanesinden MLPClassifier sınıfı kullanılarak bir model oluşturulmuştur. Bu model şu parametrelerle yapılandırılmıştır:
- **Gizli katman boyutu**: 6 nöron
- **Aktivasyon fonksiyonu**: ReLU
- **Çözücü**: SGD (Stokastik Gradyan İnişi)
- **Alfa (L2 regularizasyon)**: 0.0001
- **Batch boyutu**: 1
- **Öğrenme oranı**: Sabit (0.003)
- **Maksimum iterasyon**: 800
- **Erken durdurma**: Etkin
- **Doğrulama oranı**: 0.1
- **Değişiklik olmadan iterasyon sayısı**: 10

### 2.4 Eğitim ve Değerlendirme
Tüm modeller için:
- Veri seti %80 eğitim, %20 test olarak ayrılmıştır
- Rastgele durum (random_state) karşılaştırılabilirlik için 42 olarak sabitlenmiştir
- Tabakalı örnekleme (stratified sampling) sınıf dağılımının korunması için kullanılmıştır
- Modellerin performansı doğruluk (accuracy) metriği ile değerlendirilmiştir
- İki ve üç katmanlı modeller için, iterasyon sayısının etkisini görmek amacıyla 100'den 1000'e kadar 100'er artışlarla farklı iterasyon sayıları denenmiştir

## 3. Sonuçlar (Results)
İki ve üç katmanlı MLP modelleri için farklı iterasyon sayısı değerlerinde elde edilen doğruluk oranları tablolar halinde kaydedilmiştir. Scikit-learn MLPClassifier modeli için ise doğruluk, kesinlik, duyarlılık ve F1 skoru hesaplanmıştır. Sonuçlar incelendiğinde iki katmanlı modelin üç katmanlıya göre bu veri seti özelinde daha iyi sonuç gösterdiği görülmüştür. iki katmanlı model 800. iterasyondan sonra %90 doğruluk skoru üstüne çıkarken üç katmanlı model 900. iterasyondan sonra %90 üzerine çıkmıştır. 

İki katmanlı MLP modelinin iterasyon sayısına göre doğruluk oranları:
- İterasyonlar arttıkça doğruluk oranı genellikle artmıştır
- En iyi performans genellikle 500-1000 iterasyon aralığında gözlemlenmiştir

Üç katmanlı MLP modelinin iterasyon sayısına göre doğruluk oranları:
- İki katmanlı modele benzer şekilde, iterasyonlar arttıkça doğruluk oranı genellikle artmıştır
- Üç katmanlı model, iki katmanlı modele göre bazı iterasyon değerlerinde daha iyi performans göstermiştir

Scikit-learn MLPClassifier modeli:
- Doğruluk: Yaklaşık %99
- Kesinlik, duyarlılık ve F1 skoru değerleri de hesaplanmıştır
- Karmaşıklık matrisi (confusion matrix) ve sınıflandırma raporu (classification report) ayrıntılı değerlendirme için oluşturulmuştur

## 4. Tartışma (Discussion)
Bu çalışmada üç farklı MLP modeli karşılaştırılmış ve tümünün banka notlarının doğrulanmasında yüksek doğruluk oranları elde ettiği gözlemlenmiştir. Modellerin performansına ilişkin bazı önemli bulgular:

1. Basit bir iki katmanlı MLP bile bu sınıflandırma problemi için yüksek doğruluk sağlamaktadır, bu da problemin doğrusal olarak ayrılabilir olduğunu düşündürmektedir.

2. Üç katmanlı MLP, bazı iterasyon değerlerinde iki katmanlı modele göre daha iyi performans göstermiştir, ancak fark genellikle küçüktür. Bu, ekstra gizli katmanın karmaşık örüntüleri yakalamada yardımcı olduğunu gösterebilir.

3. Scikit-learn MLPClassifier'ın erken durdurma, mini-batch öğrenme ve diğer optimizasyonlar sayesinde manuel olarak kodlanmış modellerden daha iyi performans gösterme potansiyeli vardır.

4. İterasyon sayısı arttıkça modellerin doğruluk oranları genellikle artmıştır, ancak belirli bir noktadan sonra iyileşme marjinal hale gelmiştir. Bu, eğitim süresini optimize etmek için bir denge noktası bulmak gerektiğini göstermektedir.

5. ReLU aktivasyon fonksiyonu ve He başlatması kullanımı, vanishing gradient problemini azaltarak modellerin etkili bir şekilde eğitilmesine katkıda bulunmuştur.

## 5. Referanslar (References)
1. Haykin, S. (2009). Neural networks and learning machines (Vol. 3). Upper Saddle River, NJ, USA: Pearson.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
3. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12, 2825-2830.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
5. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

