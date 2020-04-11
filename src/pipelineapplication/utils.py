import numpy as np
import matplotlib.pyplot as plt


def get_abstracts_to_analyze():
    """
    This is only a simple function that helps to keep the abstract to analyze
    :return: list of dictionaries
    """
    abstracts = [{
        "id": "45229eae3050b2f0379efbd6e6caff2991445c5a",
        "title": "Accelerating Numerical Dense Linear Algebra Calculations with GPUs",
        "year": "2014",
        "abstract": "This chapter presents the current best design and implementation practices for the acceleration "
                    "of dense linear algebra (DLA) on GPUs. Examples are given with fundamental algorithms—from the "
                    "matrix–matrix multiplication kernel written in CUDA to the higher level algorithms for solving "
                    "linear systems, eigenvalue and SVD problems. The implementations are available through the "
                    "MAGMA library—a redesign for GPUs of the popular LAPACK. To generate the extreme level of "
                    "parallelism needed for the efficient use of GPUs, algorithms of interest are redesigned and "
                    "then split into well-chosen computational tasks. The tasks execution is scheduled over the "
                    "computational components of a hybrid system of multicore CPUs with GPU accelerators using either "
                    "static scheduling or a light-weight runtime system. The use of light-weight runtime systems "
                    "keeps scheduling overhead low, similar to static scheduling, while enabling the expression of "
                    "parallelism through sequential-like code. This simplifies the development effort and allows "
                    "the exploration of the unique strengths of the various hardware components.",
        "outCitations": "984f46db549196e9e8f6f6a68de9a5e895d0b71e, c710e49026459afae66a332df6ef9c08049ca1a1, "
                        "41742820c229f191f572037b74fc0f61cc873f97, 2b22f17c6975064805dfd4170d947c24e1282b3c, "
                        "96e052147534e27e5d75d448af0504b597d6544c, 385676f16146c079c1a2fbdc4e69a4547440f53f, "
                        "dc436daa42017bf5a56348fc571b3a9f88dc9e80, a092a4ea4e161be82ecbaf1a7f631e860c8a3c5a, "
                        "0a5617cf569abe3c669a71f4c604d47ca334ae12, a5d594b6979462aa871de08e7c887b1058e4c2b7, "
                        "5b458ee989583e9a6a67046ed9441e254a49b1da, e326e4f9c2a8712fd50a46b0c29df729a6232cce, "
                        "ff5f5806032927b94698a83cacc9c1caeafd8ade, 3f4cc3f123c9ae66413b686675cc239389612e15"
    }, {
        "id": "7972d64ceb58cc8ab5a439f52d1d137be640a54b",
        "title": "Video-based surgical skill assessment using 3D convolutional neural networks",
        "year": "2019",
        "abstract": "PurposeA profound education of novice surgeons is crucial to ensure that surgical interventions "
                    "are effective and safe. One important aspect is the teaching of technical skills for minimally "
                    "invasive or robot-assisted procedures. This includes the objective and preferably automatic "
                    "assessment of surgical skill. Recent studies presented good results for automatic, objective "
                    "skill evaluation by collecting and analyzing motion data such as trajectories of surgical "
                    "instruments. However, obtaining the motion data generally requires additional equipment for "
                    "instrument tracking or the availability of a robotic surgery system to capture kinematic data. "
                    "In contrast, we investigate a method for automatic, objective skill assessment that requires "
                    "video data only. This has the advantage that video can be collected effortlessly during minimally "
                    "invasive and robot-assisted training scenarios.MethodsOur method builds on recent advances "
                    "in deep learning-based video classification. Specifically, we propose to use an inflated 3D "
                    "ConvNet to classify snippets, i.e., stacks of a few consecutive frames, extracted from surgical "
                    "video. The network is extended into a temporal segment network during training.ResultsWe "
                    "evaluate the method on the publicly available JIGSAWS dataset, which consists of recordings "
                    "of basic robot-assisted surgery tasks performed on a dry lab bench-top model. Our approach "
                    "achieves high skill classification accuracies ranging from 95.1 to 100.0%.ConclusionsOur results "
                    "demonstrate the feasibility of deep learning-based assessment of technical skill from surgical "
                    "video. Notably, the 3D ConvNet is able to learn meaningful patterns directly from the data, "
                    "alleviating the need for manual feature engineering. Further evaluation will require more "
                    "annotated data for training and testing.",
        "outCitations": "984f46db549196e9e8f6f6a68de9a5e895d0b71e, c710e49026459afae66a332df6ef9c08049ca1a1, "
                        "41742820c229f191f572037b74fc0f61cc873f97, 2b22f17c6975064805dfd4170d947c24e1282b3c, "
                        "96e052147534e27e5d75d448af0504b597d6544c, 385676f16146c079c1a2fbdc4e69a4547440f53f, "
                        "dc436daa42017bf5a56348fc571b3a9f88dc9e80, a092a4ea4e161be82ecbaf1a7f631e860c8a3c5a, "
                        "0a5617cf569abe3c669a71f4c604d47ca334ae12, a5d594b6979462aa871de08e7c887b1058e4c2b7, "
                        "5b458ee989583e9a6a67046ed9441e254a49b1da, e326e4f9c2a8712fd50a46b0c29df729a6232cce, "
                        "ff5f5806032927b94698a83cacc9c1caeafd8ade, 3f4cc3f123c9ae66413b686675cc239389612e15"
    }, {
        "id": "d644d4eb388885fe651257ad234b1cb9bb1dc0d4",
        "title": "Tensor Restricted Isometry Property Analysis For a Large Class of Random Measurement Ensembles",
        "year": "2019",
        "abstract": "In previous work, theoretical analysis based on the tensor Restricted Isometry Property (t-RIP) "
                    "established the robust recovery guarantees of a low-tubal-rank tensor. The obtained sufficient "
                    "conditions depend strongly on the assumption that the linear measurement maps satisfy the t-RIP. "
                    "In this paper, by exploiting the probabilistic arguments, we prove that such linear measurement "
                    "maps exist under suitable conditions on the number of measurements in terms of the tubal rank r "
                    "and the size of third-order tensor n1, n2, n3. And the obtained minimal possible number of "
                    "linear measurements is nearly optimal compared with the degrees of freedom of a tensor with "
                    "tubal rank r. Specially, we consider a random sub-Gaussian distribution that includes Gaussian, "
                    "Bernoulli and all bounded distributions and construct a large class of linear maps that satisfy "
                    "a t-RIP with high probability. Moreover, the validity of the required number of measurements is "
                    "verified by numerical experiments.",
        "outCitations": "b5e853572b2f3134acafa76d5ae80b9f28c7dca8, 0c9bb579d8ad6ac987f7a16b66ddace671fc57c5, "
                        "6407ac1f051d30ca621ce16cf4ca67beb05930c6, 4f143cbc63e4c202db77566dac0f1f08c0774a45, "
                        "9fb8c76e6b17f3fdbd0c8f293ce8da4b79f4ffeb, ca96af5be9e713206b92e64f7824da1c2e38e4d4, "
                        "4f9f19e11baa16cb7ae27752864cb4231109b14c, 25ffa8c55f509241a577a6e6ec85792c66d1f41a, "
                        "ea1cf46db3f99aa538d8397e2d3cc9947213f0ad, 9336bfdcb40532ffd0d890854fee6fd3d98254ac, "
                        "7463c834bc397b6c4c2b1fd5e121b0f5007227cf, 6661789de63b3cebe2eafdd7e9e7a316ad1f0b8f, "
                        "81e8556180e4410b77c4cac9844f3547df97a345, 105858b27b052bd7f1e433f176fe8b4eb3f1ca84, "
                        "b1827bb4ee21b5e6b196b3f6995a9fa6928f716c, a07a7ea0cad9eb6e02fe8731c9f16c5115a16203, "
                        "e24c0387fa0ec3c32ebc97c050c94b8b01eeadd2, a57b6a19b1eef9228a034f61ba47022844cca042, "
                        "2ddc0494377c7cb162286cf6c5664ae901cda588, 5c2e30085a1cd03a36ca84c102eff54f9d923ad1, "
                        "fd37f013f36d9a0869a125499d7e9be1fcc3c1b3, 78e36cfa36220038ea3d79d3b0cc689238074e51, "
                        "2c2acfdbd61336f0b4d9974a7ff6ce963f6a73e5, 6179b2232c635008fed0f5ea0c8c5c82ccd1bdda, "
                        "f42f8d0b8e331255bcfe891e4ecebb5260b761c2, ae1483451dd70769ccbbf199d99cc553daa19852, "
                        "66f3f64b0a18b54642390dddbc2264093c63e3e3, 658339c73021674da17354eaaffc3cd326332117, "
                        "f6047e62635ebf241e3c9e6697d4fb39a098379f"
    }, {
        "id": "1238b5b846564ac04d4bb22df11c2b5cbf9680fd",
        "title": "Compressed Sensing Reconstruction of Hyperspectral Images Based on Spectral Unmixing",
        "year": "2018",
        "abstract": "How to utilize the characteristics of hyperspectral images (HSIs) is a key problem in application "
                    "of compressed sensing theory to hyperspectral image compression and reconstruction. Based on "
                    "the study of spectral mixing characteristics, a compressed sensing reconstruction algorithm "
                    "with spectral unmixing for HSIs is proposed. Taking advantage of linear mixing model, the HSIs "
                    "are separated into endmember matrix and abundance matrix. Instead of directly reconstructing "
                    "the entire hyperspectral data as traditional reconstruction algorithms, the proposed algorithm "
                    "explores the idea of spectral unmixing for reconstruction. In the sampling process, the HSIs "
                    "are sampled both spatially and spectrally. In the reconstruction process, a joint optimization "
                    "problem for endmember extraction and abundance estimation is established and solved in an "
                    "iterative way to obtain the reconstructed hyperspectral data. Experimental results on synthetic "
                    "and real hyperspectral data demonstrate that the proposed algorithm could obtain the endmember "
                    "and abundance information effectively, and the accuracy of reconstructed HSIs as well as the "
                    "computational efficiency are superior to the state-of-the-art reconstruction algorithms.",
        "outCitations": "25f7a27a2c8bb01df6937952645844ccc4dff89e, 366f9b38cd4a8834d196b2f5d458fefac95e3280, "
                        "17282e51627d7064b59cd9442f86f213ce61b7b3, cc79e154ee8cf75e8d132f497b64f7c10c380bcd, "
                        "1a9e6159b34612df15dd56b87780d96e55d1b878, 8f55a1caee01e7280cc3e41c4d6a2b348ff1f69e, "
                        "57b10f9f7191bc198731d97be06eee98c62a08ac, 93f75219cf6f5431db450601429b299a303e9443, "
                        "a898ad13c96e5c068a2e4fc88227278e646b712e, bd91ecaf8421f83885b35803f15688b50c2ae965, "
                        "690ea5048da479d32a5319d01b6775f925e10031, 468f1f3a3bbf1719adcae61088215aafeb6eab85, "
                        "b830ba166a6c56706ae6299792f0894e4f36ca12, 97dbe219299220fb8dfc1adc1b16453cf8245e0b, "
                        "1a4b96e71b72af9d140b4f7a4f0fb027a592d4d7, 6ad6afe56f793a3a4facaa9e2dd2ec3622f1940b, "
                        "5334790f36aa7ab9e7fcd0739564ceca391a21f0, 0eb7e13acaf8a8e3e701a0336ee7824fb131fdb3, "
                        "e50f0b7976ccf1a975f8ca32ac40cd512c495406, 2cc257b0c7db92f90c3224c35df7b8e85f57a090, "
                        "117579452885b0f42315a880067dd39ff4eac7c5, 42c41a849b9c439371b93ea0b7e44230f668679f, "
                        "28b004d775a55f8f8ca1c0c4c638ec3ca03fe673, d51d5779984ab2da2c87d29ad2c08ada7289f186, "
                        "306707406fd6ffb877c4d43b949e05950b268291, cd9d7e26fb2184b387ca1de3c6ad0f31fdb996b5, "
                        "aa01785ab55fedd5ba56dbeb19a99213086b292e, ee4815654d511eb0b6064e94d09aa1ce3f64a3c2, "
                        "2118ee75afd96adbe0de21b51166f8815e5b590b, eae2e0fa72e898c289365c0af16daf57a7a6cf40, "
                        "faa870e2f5a536894a49acfbfe980008f52a1f9d, 3ddff2769afc96d0d99b79f6a708da0d46f04c5b, "
                        "0dd91b26101cbb92cc92a1ec84b70ac86419caa0, 4fa102f339c9974532ceb1dcb63e8ece66ea1e42, "
                        "88a0dccc6b90dbbc9daf984da4e70ceecb3ef891, ee3d02a5a5d39d36d66fd416e6af204b44d7ba1a, "
                        "85a52e5e349c5874297b82a42d6d888ac03610cd, c0a98da299e5b273053527f9ef8d1974b9d64fac, "
                        "45f2bd821d663e07d11bf8d4371a7fb4f0fa5e2c, 7dfce7cd85255d25b7dbf9e41d3321a3b5f817e0"
    }, {
        "id": "a82bfd6530fda11d11d90aee16baac506bc2001d",
        "title": "Content Driven Enrichment of Formal Text using Concept Definitions and Applications",
        "year": "2018",
        "abstract": 'Formal text is objective, unambiguous and tends to have complex sentence construction intended '
                    'to be understood by the target demographic. However, in the absence of domain knowledge it is '
                    'imperative to define key concepts and their relationship in the text for correct interpretation '
                    'for general readers. To address this, we propose a text enrichment framework that identifies '
                    'the key concepts from input text, highlights definitions and fetches the definition from external '
                    'data sources in case the concept is undefined. Beyond concept definitions, the system enriches '
                    'the input text with concept applications and a pre-requisite concept graph that showcases the '
                    'inter-dependency within the extracted concepts. While the problem of learning definition '
                    'statements is attempted in literature, the task of learning application statements is novel. '
                    'We manually annotated a dataset for training a deep learning network for identifying application '
                    'statements in text. We quantitatively compared the results of both application and definition '
                    'identification models with standard baselines. To validate the utility of the proposed framework '
                    'for general readers, we report enrichment accuracy and show promising results.',
        "outCitations": "fbb5d9a795935a5efe2ebfa2a013d0160cf5bdf0, f37e1b62a767a307c046404ca96bc140b3e68cb5, "
                        "4ed75f35ae7343cd906f311390f572a58e36805c, 6e854c27ebb9625bc409469541f21526b0df7bd5, "
                        "cb6be7b2eb8382a85fdc48f1ca123d59d7b003ce, 1355fda225ce00c93123a6e41f6777e2270038b5, "
                        "d333fdfcc56483336f0428e48bc11cd26cac28d8, e626e53adaf80109486563a98b87dc21f0652318, "
                        "3e48cb9b4838f8947388c3926df70211ca49b51a, a14045a751f5d8ed387c8630a86a3a2861b90643, "
                        "c8961c424778ef9b15008d45b314c2d319a7cc56, 79f21f3dc4e01c1830b3d7f5cf10e170dbc948c1, "
                        "1ddfef3dd1e7e061e011b86a0f62b259903bc55b, eb42a490cf4f186d3383c92963817d100afd81e2, "
                        "4bb202bb748c0447121a4726ee2e521196930bb2"
    }, {
        "id": "d9801dbd066d104d244df6aace24fd56e7e92f3d",
        "title": "Linear demixed domain multichannel nonnegative matrix factorization for speech enhancement",
        "year": "2017",
        "abstract": 'In this paper, we investigate blind source separation for audio signals based on multichannel '
                    'nonnegative matrix factorization (MNMF) of magnitude spectrograms in a linear demixed domain. '
                    'The original magnitude MNMF by itself is less effective in general acoustic situations because '
                    'it discards mutual information between input channels, which is represented by non-diagonal '
                    'complex elements of the spatial covariance matrices of them. To deal with this problem, several '
                    'linear transformations of the multichannel input have been proposed in order to diagonalize '
                    'the covariance matrices without loss of the mutual information. However, when the number of '
                    'microphones is small, it is difficult for static transformations to work well for various '
                    'combinations of source positions. For this problem, we first prove that general linear '
                    'transformations (linear demixing) can be applied as preprocessing of the magnitude MNMF, and '
                    'then confirm that a transformation adaptive to source positions, such as using frequency domain '
                    'independent component analysis, is better than the conventional static transformation by '
                    'experimental comparison of 2- and 4-channel noisy speech enhancement tasks.',
        "outCitations": "01a7cfc4721b85ab15304644db5235f334a6aa1e, 2bfaa14d9774ff2fbb03327d291b2c64dc13a6b1, "
                        "72d175c4a0b2cab7b6b0912c8b98bfac002072de, 8d39d0853d88f92714684931a5cc1f2fa9eee11c, "
                        "29bae9472203546847ec1352a604566d0f602728, a4e79f0316ffafc2610f085e09cd3fea9cd41bfd, "
                        "d3c2095c74ece80a85a6210ad1e6263e47afaacb, 89e07527f8b1be40be563001e2ab614721a2c7fd, "
                        "7e745ef5cb1e19c129ccb7994f51f10dd2e3645d, 783f44598d5d9df7d534d00943959d9b34dbc302, "
                        "d4660c13cfeb749896b01e14cae819458793fccb, 77eac533c56437e88e09a45a3b254d5907c6f1ed, "
                        "8338daf3b445fb1b65a203bc60213c09cb537585, be7feda578a13b2c4f44a896798b24de581c137a, "
                        "29de8281b8cbc764d605a20d00b818eba6d47da1, e99db5a3dcc3da3cdffc01c8bbd656aca1112f69, "
                        "531a4f5446a13a5dbdfe3e22340d4c3fd6f6a671"
    }, {
        "id": "eb8aa8b78bb51773c4fad9304ca50b3452e43f19",
        "title": "Opinion Mining and Visualization of Online Users Reviews: A Case Study in Booking.com",
        "year": "2018",
        "abstract": 'The growth of web applications and portals on hotel booking have led to an enormous amount of '
                    'consumer generated comments and reviews on various hotels and travel services. In this paper, '
                    'we present a work on the automatic analysis of user reviews on online hotel reviews with the '
                    'aim to understand public opinions towards facilities and services they receive. We follow an '
                    'aspect based approach where initially Latent Dirichlet Allocation is utilized to model topic '
                    'opinions. The aspects specified indicate the important characteristics of the services and the '
                    'facilities that users address in their reviews. After that, natural language processing '
                    'approaches are used to analyze textual reviews, specify the dependencies on a sentence level '
                    'and assist in understanding users opinions. At the final stage of understanding users opinions, '
                    'several classifiers are trained under different feature sets extracted from the textual reviews '
                    'and combined in ensemble schemas and their performance is examined on recognizing the polarity '
                    'of the users opinions. The results are quite satisfactory and indicate that features such as '
                    'sentence dependencies assisted the classifiers in achieving accurate performance and that '
                    'ensemble schemas perform robustly better than individual classifiers.',
        "outCitations": "a0456c27cdd58f197032c1c8b4f304f09d4c9bc5, 7a92314ff9ef62aab0f30510225736aa30aa7b84, "
                        "3b67c9b6c3328ead7dc123c92ee0e5f2c31317cb, 805021df356a8e4fbabd9779e88e1b09576c7399, "
                        "08281550b59e942475baee8c8c7a27e4246a8f7e, f39e6d1cfd83ab9529367c1bb694abfc9d331b3d, "
                        "c1dfcc98622dfd22480e4028bdc0244713e449d7, be10b7485d76ea04d1982082ec3f253f3e322e34, "
                        "7373725d5a84669a93784b0608a363162e449642, 58f7accbe36aadd0ef83fd2746c879079eb816bc, "
                        "9e7f8d1777f2e5dfc216eb0ee6ca30ae58f686fd, 3cc228402f31ca749112197720b9ef6af0c16790, "
                        "984585dedd6c3907063d04f94ac9c385be633641, 9afe81e3a498bc31cb749e7d7972ff319a730961, "
                        "fca5d9167b5cc43f225a3295cba3379b97e1b5dc, 2f5102ec3f70d0dea98c957cc2cab4d15d83a2da, "
                        "961f247287f844ea9710bca5acd92bd3a8559ab5, 5c695f1810951ad1bbdf7da5f736790dca240e5b, "
                        "2671e9ff0eb68739d6610cf67937e6b34a2fcf10, 88e055b05be080154c901f755bf4381f75250948"
    }, {
        "id": "89518a0e00c5c81ff8be8ab82d4da2b7049a4306",
        "title": "FMLLR Speaker Normalization With i-Vector: In Pseudo-FMLLR and Distillation Framework",
        "year": "2018",
        "abstract": 'When an automatic speech recognition (ASR) system is deployed for real-world applications, '
                    'it often receives only one utterance at a time for decoding. This single utterance could be '
                    'of short duration depending on the ASR task. In these cases, robust estimation of speaker '
                    'normalizing methods like feature-space maximum likelihood linear regression (FMLLR) and i-vectors '
                    'may not be feasible. In this paper, we propose two unsupervised speaker normalization '
                    'techniques—one at feature level and other at model level of acoustic modeling—to overcome '
                    'the drawbacks of FMLLR and i-vectors in real-time scenarios. At feature level, we propose the '
                    'use of deep neural networks (DNN) to generate pseudo-FMLLR features from time-synchronous pair '
                    'of filterbank and FMLLR features. These pseudo-FMLLR features can then be used for DNN acoustic '
                    'model training and decoding. At model level, we propose a generalized distillation framework, '
                    'where a teacher DNN trained on FMLLR features guides the training and optimization of a student '
                    'DNN trained on filterbank features. In both the proposed methods, the ambiguity in choosing the '
                    'speaker-specific FMLLR transform can be reduced by augmenting i-vectors to the input filterbank '
                    'features. Experiments conducted on 33-h and 110-h subsets of Switchboard corpus show that the '
                    'proposed methods provide significant gains over DNNs trained on FMLLR, i-vector appended FMLLR, '
                    'filterbank and i -vector appended filterbank features, in real-time scenario.',
        "outCitations": "177496ecd8b485679d00839a1e19dadcab1f802e, 4d6e574c76e4a5ebbdb5f6e382d06c058090e4b7, "
                        "d233a365e9ae0ba7d86d1b0b0a2b9e66983bf551, cb4d59854786c57e3da4ca2a433b826e2f7a2ef4, "
                        "f317b89d3224b6bc5269790e81604904cff7052c, c256a54a5f3f07a6dcf2dea3a220d0024cf3bfe5, "
                        "2978880d0c3a469a1420411d4b0b30a7b3fe56e9, 7b4ee51aa8581889d131f35167e8640568fdde98, "
                        "8660642c37be1eaeca0d22598558249ac47d767d, 5b73ba929d4d8e7eb7c6a96703679464b3208538, "
                        "806da4fd62dadf78c8f26a4ec57529ffacf15316, d237de6e4974e6a34d2b35d7a3a223f6fb611219, "
                        "d8b89cf577350095c8b52555e5482e96e6e207fc, 2904a5c43940a2d1da84d4c4a387cb17de987ffb, "
                        "7599dfed1de67c726f9e4fd372cc9ef03d2cf3e9, 58059409e131f2a854367052636138e835f14f60, "
                        "bbf8e55f0837c08688106a30fd2560f39327f54b, 94de64f3a47e5b06cb0960ddf68a7fbc1ac70232, "
                        "0670e66badee8a9772bb326721e5fe5045817303, 7600cf5da33b19d37266da4d2edcbd32bc0ecb5e, "
                        "f341fa61da527e64a349334836d52626fe9d6c79, d3bdc4a679cafff12901039245b99411d5b729f7, "
                        "0c908739fbff75f03469d13d4a1a07de3414ee19, 8e46a2e57ce37b846bef48d776aeafa16c411681, "
                        "3a1a2cff2b70fb84a7ca7d97f8adcc5855851795"
    }
    ]

    for elem in abstracts:
        elem["outCitations"] = elem["outCitations"].split(", ")

    return abstracts


def normalize_scores_on_cittopics(cittopic, p):
    """
    This function is used to normalize scores in a specific CitTopic
    :param cittopic: the CitTopic score list
    :param p: max score to assigned to a topic
    :return: the normalized CitTopic score list
    """
    max_value = p * len(cittopic) + 1

    return list(map(lambda x: np.round(x / max_value, 3), cittopic))


def compute_missing_citations(cittopics, outcitations):
    """
    This function is used to compute the missing citations for a paper
    :param cittopics: the CitTopic
    :param outcitations: the list of records including outcitations
    :return: a list of missing citations
    """
    missing = []
    for c in cittopics:
        if c not in outcitations:
            missing.append(c)

    return missing


def compute_hit_citations(cittopics, outcitations):
    """
    This function is used to compute the missing citations for a paper
    :param cittopics: the CitTopic
    :param outcitations: the list of records including outcitations
    :return: a list of missing citations
    """
    hit = []
    for c in cittopics:
        if c in outcitations:
            hit.append(c)

    return hit


def get_valid_predictions(predictions, t):
    """
    This function is used to get all valid prediction taking into account a input threshold for probability
    :param t: threshold
    :param predictions: list prediction probability
    :return:
    """
    predictions = list(enumerate(predictions))
    res = list(filter(lambda x: x[1] > t, predictions))

    return res


def make_bar_plot(height, bars, title, path):
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title(title)
    plt.savefig(path)
    plt.close()


def split_list(l, p):
    n = (100 / p)
    k = len(l) / n

    if k % 1 > 0:
        k = int(k) + 1
    else:
        k = int(k)

    return [l[i:i + k] for i in range(0, len(l), k)]
