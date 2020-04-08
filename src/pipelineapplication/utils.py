import numpy as np


def get_abstracts_to_analyze():
    """
    This is only a simple function that helps to keep the abstract to analyze
    :return: list of dictionaries
    """
    abstracts = [{
        "id": "20c507d11b20b2abe4c15a786653ba4fa1a6479a",
        "title": "A weighted multi-granulation decision-theoretic approach to multi-source decision systems",
        "year": "2017",
        "abstract": "Decision theoretic rough set is a typical generalization model of rough set, which has fault "
                    "tolerance based on Bayes minimum decision risk. How to mine knowledge from the information "
                    "collected from different sources is one of the focuses of current artificial intelligence. "
                    "From a cognitive point of view, especially from the point of granulation, this paper studies "
                    "decision theory of multi-source decision systems based on generalized multi-granulation and "
                    "decision theoretic rough sets. It is because each granular structure is not equally important "
                    "in practical issues. First of all, the method of granulation weight is proposed based on the "
                    "internal uncertainty of systems and the external correlation between systems, namely the double "
                    "weighted granulation (DGW) method. And then a weighted generalized multi-granulation "
                    "decision-theoretic rough set (WGM-DTRS) model in multi-source decision systems is proposed. "
                    "Finally, in order to verify the effectiveness of the (DGW) method, the approximation accuracy "
                    "of decision classes under different weighted granulation methods is compared. The numerical "
                    "results show that the proposed method is effective from the point of classification. Therefore, "
                    "the WGM-DTRS model based on the DGW method is meaningful.",
        "outCitations": "32b3c9f0ef19cc33a38e893ff16dff1d85b7df72, 7f1dec3dcf4abacb9a593d24c85abef5ecfd0e30, "
                        "33711930066501cd6efb95ef5125c3f9bb42d885, cdef00d237dc8f9bcbd16d02be429ddf6f10121b, "
                        "eae32037595b6878c13a0466b026bd9ac9a645f5, 6cef63321eda6256de72f23e64ade797c11c8121, "
                        "26029ee7e87a48d57081b087cd18680be942b9f8"
    }, {
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
        "id": "3980e764f677dd67181f68e47710e53370584800",
        "title": "On topological structure of web services networks for composition",
        "year": "2013",
        "abstract": 'In order to deal efficiently with the exponential growth of the web services landscape '
                    'in composition life cycle activities, it is necessary to have a clear view of its main '
                    'features. As for many situations where there is a lot of interacting entities, the complex '
                    'networks paradigm is an appropriate approach to analyse the interactions between the multitudes '
                    'of web services. In this paper, we present and investigate the main interactions between Semantic '
                    'Web services models from the complex network perspective. Results show that both parameter '
                    'and operation networks exhibit the main characteristics of typical real-world complex networks '
                    'such as the "small-world" property and an inhomogeneous degree distribution. These results yield '
                    'valuable insight in order to develop composition search algorithms, to deal with security threat '
                    'in the composition process and on the phenomena which characterise its evolution.',
        "outCitations": "bf4ae2c4e258bdd9d21562b5c05420033a02cc1d, da4a0c1eebdec2893e1adc41925a09a249cf7bc4, "
                        "9da2906c0aaec6c50fbe272a97a4888bdffe85b4, 449030fac9f669e668bbed43376e7cb642fc3634, "
                        "65b5b6de48a768526f16ea98fddfa6e4e70af8c1, 4c6b395361f12af51ccd188f4f8d6b55f811ecf0, "
                        "08ef5800c4cf9f8fc74c8332bbd83036a7b1643e, c27f231edae7f502999e3844dc383acd5a91fa26, "
                        "8c893e64e33eaad6e8013fdae1bcbe6ee757e020, c75da49837a66d90b014cfc8b923946e409d01b6, "
                        "a38b09f352aab3cb58dc212a2c77508a91ac4558, cb15c215c2dfb94931a605ea69ca7d01cbb4f180, "
                        "9be428c9383d47b86570b1b9fc20faf006346c5d, e876229734551231d98059a174a3381b0013d833, "
                        "a98736e2e9dd0c4870bc40aa083d0baa534a5cc5, 3b9b9c57ad7431af5524dfb1d0b044c19cfcd2e8, "
                        "f83dcce3b749272062362af70dac675632b56ffb, c67dcd5066c7f72c71a0d14ffef7206d14ddd270, "
                        "afdcba53013b26012afd1166fd7419aa232d13b6, 265be00bf112c6cb2fa3e8176bff8394a114dbde, "
                        "8c38fe60b3761e25ed2d2486c3a944674a3e9534, 7d32e764672117c2d5bce2c69dee737c6e88b719, "
                        "db04bb965705706a844c7ac496f79a25842408e8, cdec9816d3aa68b9aa3b23858e265ced85fa040b, "
                        "6b89e8eb15a572b80cefc2fc21c67d71dd979ee4, bc2f9329c3260e82b0a39f7a725c03e7512b40f7, "
                        "8e0870ef4b6e000df65dec26b9d5873cfd444041, 195deb7f86b027db97fd47f4fee13f8e11262a6c, "
                        "45d9a0db23906842f4c3385e4f231ede34d32db0, f2dde5e984592007f6e532561ef091e6dd8fba2c, "
                        "4cb0526227866e895dcce2e1acd1fa5701b797cb, 45bbc03cd39ea1cb6287e7d8f5f12fa363817f5c"
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
    res = []
    for i in range(len(predictions)):
        if predictions[i] > t:
            res.append((i, predictions[i]))

    return res
