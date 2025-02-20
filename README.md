# FaceRecognizer
Face Recognizer

<p>Real-Time Face Detection and Recognition for Enhanced SecurityApplicationsCSC-180, Section 01, Group 11Taekjin Jungtaekjinjung@csus.eduCalifornia State UniversitySacramentoSacramento, California, USAPriyatham Dasigandhapriyathamdasigandha@csus.eduCalifornia State UniversitySacramentoSacramento, California, USAJenil Bhupathi Shingalajshingala@csus.eduCalifornia State UniversitySacramentoSacramento, California, USAAbstractThis paper presents an intelligent face recognition system designedto enhance security and automation in real-world applications. Oursystem combines YOLOv8 for face detection with a custom CNNarchitecture based on InceptionV3 for face recognition. The systemaddresses critical challenges in modern security systems by provid-ing real-time detection and classification of individuals, achieving100% accuracy in unknown face detection while maintaining robustperformance for known individuals. Our implementation demon-strates particular effectiveness in both security monitoring andautomated attendance applications, making it suitable for variousreal-world scenarios including home security and educational envi-ronments.CCS Concepts• Computer systems organization →Neural networks.Keywordsface recognition, deep learning, CNN, YOLO, security systems, bio-metricsACM Reference Format:Taekjin Jung, Priyatham Dasigandha, and Jenil Bhupathi Shingala. 2024.Real-Time Face Detection and Recognition for Enhanced Security Applica-tions: CSC-180, Section 01, Group 11. In . ACM, New York, NY, USA, 5 pages.https://doi.org/10.1145/nnnnnnn.nnnnnnn1Introduction1.1Problem DescriptionIn today’s world, security threats and time-consuming manual iden-tification tasks pose significant challenges. There is a growing needfor automated systems that can provide continuous monitoringand real-time threat detection. Traditional security systems oftenrequire constant human monitoring, which is both inefficient andprone to human error. Our face recognition system addresses thesechallenges by:Permission to make digital or hard copies of all or part of this work for personal orclassroom use is granted without fee provided that copies are not made or distributedfor profit or commercial advantage and that copies bear this notice and the full citationon the first page. Copyrights for components of this work owned by others than theauthor(s) must be honored. Abstracting with credit is permitted. To copy otherwise, orrepublish, to post on servers or to redistribute to lists, requires prior specific permissionand/or a fee. Request permissions from permissions@acm.org.Conference’17, July 2017, Washington, DC, USA© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.ACM ISBN 978-x-xxxx-xxxx-x/YY/MMhttps://doi.org/10.1145/nnnnnnn.nnnnnnn• Providing automated alert systems for detecting unknownindividuals• Enabling efficient attendance tracking in educational settings• Creating a foundation for broader security applications• Offering real-time processing capabilities1.2MotivationSeveral real-world scenarios demonstrate the necessity of our sys-tem:• Home security systems require 24/7 monitoring to protectfamily members• Educational institutions spend valuable time on manual at-tendance tracking• Security systems need quick identification of potential threats2System Architecture2.1Dataset DescriptionOur system utilizes two main datasets:• Custom dataset (151 images) of known individuals• LFW dataset containing approximately 13,000 face imagesfor unknown person detectionDataset preprocessing involves:• Image standardization to 200×200×3 resolution• Data augmentation for improved robustness• Class balancing techniques2.2System OverviewOur system employs a two-stage approach:2.2.1Face Detection Module.• YOLOv8 architecture for real-time object (face) detection• Accurate face localization under various conditions• Bounding box generation for detected facesListing 1: YOLO Face Detection Implementation# download modelmodel<em>path = hf</em>hub<em>download(repo</em>id="arnabdhar/YOLOv8 -Face -Detection",filename="model.pt")# load modelfacial<em>model = YOLO(model</em>path)def detect<em>faces(frame):Conference’17, July 2017, Washington, DC, USATaekjin Jung, Priyatham Dasigandha, and Jenil Bhupathi Shingala# Preprocess frameprocessed = cv2.resize(frame , (640, 640))processed = processed.astype('float32 ') / 255.0# Detect facesdetections = facial</em>model(processed)return post<em>process</em>detections(detections)2.2.2Face Recognition Module.• Custom CNN architecture transferring InceptionV3 layers• Feature extraction for distinctive facial characteristics• Classification using softmax activation2.2.3Face Recognition Module. Our face recognition system in-corporates the following key components:• Custom CNN architecture transferring InceptionV3 layers• Feature extraction for distinctive facial characteristics• Classification using softmax activation2.2.4Training Parameters. The model was trained with the follow-ing configuration:• Batch size: 128• Epochs: 1000• Learning rate: 0.0001• Optimizer: Adam• Early stopping patience: 102.3Data AugmentationTo improve model robustness, we implemented comprehensive dataaugmentation:Listing 2: Data Augmentation Implementationdatagen = ImageDataGenerator(rotation<em>range =20,width</em>shift<em>range =0.1,height</em>shift<em>range =0.1,shear</em>range =0.2,zoom<em>range =0.1,horizontal</em>flip=True ,brightness<em>range =[0.8, 1.2])Input 200×200×3InceptionV3Base ModelDense Layers 512-256-256Softmax OutputInputFeaturesClassificationFigure 1: Neural Network Architecture Overview3Implementation Details3.1Model ArchitectureVideo InputYOLOv8Face DetectionPreprocessingInceptionV3Feature ExtractionFCNN LayersClassificationRecognitionResultsFigure 2: System Architecture OverviewReal-Time Face Detection and Recognition for Enhanced Security ApplicationsConference’17, July 2017, Washington, DC, USAInput200x200x3InceptionV3Pre-trainedFlattenDense (512 units)BatchNorm + ReLUDense (256 units)Output (Softmax)Figure 3: Detailed Neural Network ArchitectureInput Image (200×200×3)Feature Extraction Layers(InceptionV3 Base)Dense(512)BatchNorm + Dense(256)Softmax OutputFigure 4: Neural Network Architecture OverviewListing 3: CNN Model Implementationdef build</em>model ():# Base Model: InceptionV3inception<em>model = InceptionV3(weights='imagenet ',input</em>shape =(200 , 200, 3),include<em>top=False)# Freeze early layersfor layer in inception</em>model.layers [: -5]:layer.trainable = False# Custom Architecturemodel = Sequential ([inception<em>model ,Flatten (),Dense (512, activation='relu'),BatchNormalization (),Dense (256, activation='relu'),BatchNormalization (),Dense (256, activation='relu'),Dense(num</em>classes , activation="softmax")])return model3.2Training Strategy3.2.1Loss Function. We implement focal loss to handle class im-balance:Listing 4: Focal Loss Implementationdef focal<em>loss(gamma =2.0, alpha =0.25):def focal</em>loss<em>fixed(y</em>true , y<em>pred):epsilon = tf.keras.backend.epsilon ()y</em>pred = tf.clip<em>by</em>value(y<em>pred ,epsilon ,1.0 - epsilon)cross</em>entropy = -y<em>true * tf.math.log(y</em>pred)loss = alpha * tf.math.pow(1 - y<em>pred , gamma) * cross</em>entropyreturn tf.reduce<em>mean(tf.reduce</em>sum(loss , axis =1))return focal<em>loss</em>fixed3.2.2Training Parameters.• Batch size: 128• Epochs: 1000• Learning rate: 0.0001• Optimizer: Adam• Early stopping patience: 103.3Data AugmentationTo improve model robustness, we implement comprehensive dataaugmentation:Listing 5: Data Augmentation Implementationdatagen = ImageDataGenerator(rotation<em>range =20,width</em>shift<em>range =0.1,height</em>shift<em>range =0.1,shear</em>range =0.2,zoom<em>range =0.1,horizontal</em>flip=True ,brightness_range =[0.8 , 1.2])Conference’17, July 2017, Washington, DC, USATaekjin Jung, Priyatham Dasigandha, and Jenil Bhupathi Shingala4Experimental Results4.1Performance MetricsTable 1: Recognition Performance by CategoryCategoryPrecisionRecallF1-ScoreSupportUnknown1.001.001.003298Jenil0.620.570.5914Taekjin0.880.820.8517Priyatham0.600.530.5617Accuracy0.993346Macro avg0.770.730.753346Weighted avg0.990.990.9933464.2Results Analysisboxrule314/314 - 19s - 59ms/step - accuracy: 0.9993- loss: 0.0016[0.0015765308635309339, 0.9993026256561279]Figure 5: Training Results4.3Confusion MatrixTable 2: Confusion MatrixJenilPriyathamTaekjinunknownJenilPriyathamTaekjinunknown5Challenges and Improvements5.1Current Challenges• Managing biased datasets between known and unknowncategories• Finding optimal training functions for facial recognition• Implementing real-time processing with acceptable latency• Handling varying environmental conditions5.2Improvement Plan• Advanced Architectures: Exploration of Vision Transform-ers for improved accuracy• Dataset Balance: Equalizing category distribution (100-200images per category)• Robustness: Enhanced handling of pose variations, occlu-sions, and low-quality images• Real-time Optimization: Improved processing pipeline forfaster detection6Application AreasOur system can be deployed in various scenarios:• Security Systems: Home and commercial security monitor-ing• Educational Institutions: Automated attendance tracking• Access Control: Secure area entry management• Public Safety: Threat detection in public spaces7Work DivisionTable 3: Team Member ContributionsMemberContributionsTaekjin JungFace Recognition ImplementationReal-time ProcessingHyperparameter TuningData Preprocessing and VisualizationJenil ShingalaMotion DetectionReal-time ImplementationDocumentation and ReportingPriyatham DasigandhaFace Detection (YOLO)System DebuggingTesting and Validation8Learning ExperienceThrough this project, the team gained valuable insights and exper-tise in:• Practical implementation of deep learning architectures• Handling class imbalance in real-world datasets• Real-time video processing techniques• Collaborative software development• Technical documentation and academic writing9ConclusionOur face recognition system successfully demonstrates the poten-tial of combining modern deep learning architectures for practicalsecurity applications. The system achieves:• Perfect accuracy (100%) in unknown face detection• Robust performance in real-time scenarios• Effective handling of class imbalance• Practical implementation for various security applicationsFuture work will focus on:• Implementation of transformer architectures• Enhanced dataset balancing techniques• Improved real-time processing optimization• Integration with broader security systemsarticle filecontentsnatbibReferencesHere are some references cited in the text: [5], [4], [2], [3], [1].Real-Time Face Detection and Recognition for Enhanced Security ApplicationsConference’17, July 2017, Washington, DC, USAReferences[1] Scylla [n. d.].Facial Recognition: Practical Applications for Physical Secu-rity. Scylla. https://www.scylla.ai/facial-recognition-practical-applications-for-physical-security/ Accessed: 2024-11-25.[2] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2020.Focal Loss for Dense Object Detection. IEEE Transactions on Pattern Analysis andMachine Intelligence (2020).[3] S. Reinharz. 2023. Basics of SPM December 2024. Security Industry Associa-tion. https://www.securityindustry.org/2023/03/27/facial-recognition-for-access-control-efficient-convenient-and-accurate/[4] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and ZbigniewWojna. 2016. Rethinking the Inception Architecture for Computer Vision. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition.[5] Ultralytics. 2023. YOLOv8: A Real-Time Object Detection System. (2023).</p>

