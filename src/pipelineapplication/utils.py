import numpy as np


def get_abstracts_to_analyze():
    """
    This is only a simple function that helps to keep the abstract to analyze
    :return: list of dictionaries
    """
    abstracts = [{
        "id": "5910db3edb2481b0340cb4c4e310ae9e3cf704a9",
        "title": "Neuron-Miner: An Advanced Tool for Morphological Search and Retrieval in Neuroscientific "
                 "Image Databases",
        "year": "2016",
        "abstract": "The steadily growing amounts of digital neuroscientific data demands for a reliable, systematic, "
                    "and computationally effective retrieval algorithm. In this paper, we present Neuron-Miner, "
                    "which is a tool for fast and accurate reference-based retrieval within neuron image databases. "
                    "The proposed algorithm is established upon hashing (search and retrieval) technique by "
                    "employing multiple unsupervised random trees, collectively called as Hashing Forests (HF). "
                    "The HF are trained to parse the neuromorphological space hierarchically and preserve the "
                    "inherent neuron neighborhoods while encoding with compact binary codewords. We further introduce "
                    "the inverse-coding formulation within HF to effectively mitigate pairwise neuron similarity "
                    "comparisons, thus allowing scalability to massive databases with little additional time "
                    "overhead. The proposed hashing tool has superior approximation of the true neuromorphological "
                    "neighborhood with better retrieval and ranking performance in comparison to existing "
                    "generalized hashing methods. This is exhaustively validated by quantifying the results "
                    "over 31266 neuron reconstructions from Neuromorpho.org dataset curated from 147 different "
                    "archives. We envisage that finding and ranking similar neurons through reference-based "
                    "querying via Neuron Miner would assist neuroscientists in objectively understanding the "
                    "relationship between neuronal structure and function for applications in comparative anatomy "
                    "or diagnosis.",
        "outCitations": "ff9efce2c082d9e164065b8d6455e8c8a7233a26, 5042621f887cafde28ea4409aac9e33cb0f27d5b, "
                        "61b88be1ba215f18d371c97935e2a9fce46d7672, c9f50bc2b009dd1c957e60a5e8dd138e4c8f0ecd, "
                        "73d7218caa421995d63f83f694f4bcd83f3da44d, 5ae63da5d45cb38bfed8b737747264c645f25598, "
                        "9521d040f5d9c0b7f6683b08b31f8adfc8cb3121, deafc4e6e00765a26b934490d3e9f2aacd256767, "
                        "8121ce5d553a2a8b869629ce88610ad54428be04, 581c71da74bd3baa06693cc6d0751e7c60f81bb3, "
                        "90bc77b135ac7876290d3c251f0f1818584389b6, ed3c96f79ae477b28cf9cee2250fd120846416b2, "
                        "81b7e5635241f0fae8eac9f703f604a1a0073038, 2e74388f55f2cc704c4de410578887a53a9433b0, "
                        "58d07607dd35c39fefffd373d80f2a77ab18f5c7, c4ad1a762520cb799699e3bb209748a5881a554f, "
                        "9131b527b0e8269b3b1554aecd356bab060c9b86, 60352c78989537bb5e013f5b641e4605067857fa"
    }, {
        "id": "501f45ddd59b4a2cb461219565e51dd7cd71be18",
        "title": "Strategies for Big Data Clustering",
        "year": "2014",
        "abstract": "In the paper, an overview of methods and technologies used for big data clustering is presented. "
                    "The clustering is one of the important data mining issue especially for big data analysis, "
                    "where large volume data should be grouped. Here some clustering methods are described, great "
                    "attention is paid to the k-means method and its modifications, because it still remains one "
                    "of the popular methods and is implemented in innovative technologies for big data analysis. "
                    "Neural network-based self-organizing maps and their extensions for big data clustering are "
                    "reviewed, too. Some strategies for big data clustering are also presented and discussed. It is "
                    "shown the data of which volume can be clustered in the well known data mining systems WEKA "
                    "and KNIME and when new sophisticated technologies are needed.",
        "outCitations": "f6aa3d256c994a5950fe3262f7f18083d9b40ddd, 24a85e28954871d30ebefac06b459f8c2701e7a0, "
                        "49c5f901f22bcc90bfe186bc2cfe98541fd55699, 83b88c4166a246186fd4dbbb362fc80d1939a671, "
                        "0752ae56f80af6f69ddab7db2b46161f44e03ebf, 4cd2f28abc2291d99e158cc5f9292ced1d5f1d08, "
                        "67ed7e224ae07edb85f153ecc6b2a68acd88dbfd, 7d50991b693fc23edda316fb1487f114f6cc6706, "
                        "369c4d1f50991e676dea30933a81d6d34d46430f, 17169023358dfd8c7c4e46c81450e74beb248eb8, "
                        "a62f5a1953764af0021b992b82e7fbdee9def34d, 5b04b085ca1ffb64a57dcb6711386078a7a0d531, "
                        "8b7f216d4ecd022b547837eaaccebf90224f569b, 69660235186b725c452ea5a7d8621089a713fabc, "
                        "62dc65305df004dc27814faa38d795cffd12e2ff, eaac84f995937c39918ca5cf7b4a3c050fd7bc39, "
                        "80fd362544b593bd2250e8f5f3799882fa133ca1, 514c1690f3482ac87804d4982176a81089b85a45, "
                        "9ef4d5d6503c706ba5f09ed18fe5bef2c4ef62f8, 80c983b2f36e3db461e35a5e8836d4b20b485d4f, "
                        "d94869549face41294aca52a9dd05c5f2a8e8c94, 155d81d9f764de34c7234f0e3dc37a5bd297edee, "
                        "5c8fe9a0412a078e30eb7e5eeb0068655b673e86, 5e0f9c313b093967e38b951f92535d5927c9e326, "
                        "8faad7901db9a73cacaf92ecdedbaece87d95f92, 73b25e63f21614b839e0eb766f49c6e1b4f42ca9, "
                        "f42f6efc15cb02e2895088bd4bbb8205c6d5bd06, 600ab83b9ec48148a3a0428a02c78e47dd742d61, "
                        "8856b09c032ed4f10ef8367a8f7088fbb891ec2b, 1479d27df9ae0673165e850b3ab2e987a9e2f62b, "
                        "ac8ab51a86f1a9ae74dd0e4576d1a019f5e654ed, 8123d0b1cdb8255ec27c246d64a615025f7d908a, "
                        "d7d385f45c096082812deb1623e5af2c2915b4a9, c8881f04601bba8b5f1ac6e2a2319aef54af2418, "
                        "cfcd83473e84bc74e923aef83d09d0d3f84ae2b4, 35ffee94de452a381ecade613d8fa25c33118af9"
    }, {
        "id": "d644d4eb388885fe651257ad234b1cb9bb1dc0d4",
        "title": "Tensor Restricted Isometry Property Analysis For a Large Class of Random Measurement Ensembles",
        "year": "2019",
        "abstract": "In previous work, theoretical analysis based on the tensor Restricted Isometry Property "
                    "(t-RIP) established the robust recovery guarantees of a low-tubal-rank tensor. The obtained "
                    "sufficient conditions depend strongly on the assumption that the linear measurement maps satisfy "
                    "the t-RIP. In this paper, by exploiting the probabilistic arguments, we prove that such linear "
                    "measurement maps exist under suitable conditions on the number of measurements in terms of the "
                    "tubal rank r and the size of third-order tensor n1, n2, n3. And the obtained minimal possible "
                    "number of linear measurements is nearly optimal compared with the degrees of freedom of a tensor "
                    "with tubal rank r. Specially, we consider a random sub-Gaussian distribution that includes "
                    "Gaussian, Bernoulli and all bounded distributions and construct a large class of linear maps "
                    "that satisfy a t-RIP with high probability. Moreover, the validity of the required number of "
                    "measurements is verified by numerical experiments.",
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
        "id": "4b3a5be0f2f1c02c44b9118c5fece7f5accfaad1",
        "title": "Multiple Right-Hand Side Techniques in Semi-Explicit Time Integration Methods for Transient "
                 "Eddy-Current Problems",
        "year": "2017",
        "abstract": "The spatially discretized magnetic vector potential formulation of magnetoquasistatic field "
                    "problems is transformed from an infinitely stiff differential-algebraic equation system into "
                    "a finitely stiff ordinary differential equation (ODE) system by applying a generalized Schur "
                    "complement for nonconducting parts. The ODE can be integrated in time using explicit "
                    "time-integration schemes, such as the explicit Euler method. This requires the repeated "
                    "evaluation of a pseudo-inverse of the discrete curl–curl matrix in nonconducting material "
                    "by the preconditioned conjugate gradient (PCG) method, which forms a multiple right-hand "
                    "side problem. The subspace projection extrapolation method and proper orthogonal "
                    "decomposition are compared for the computation of suitable start vectors in each time step "
                    "for the PCG method, which reduce the number of iterations and the overall computational costs.",
        "outCitations": "7d88eaa822314dba27240de4787b2d83a3637afc, 263a1e62b81b0ed8f39322683fa4a87a858b8b8c, "
                        "cf2811db47b77adfb44750b084c0164a38e70d99, 26f9e092b3f1d4d22935bc6b4a5d47f604d36060, "
                        "9b6342d6c328d5ff1e86bdfaa3ed919d205f631b, a1c897bda4c3d3c12d314f172b3c0199816d3ea7, "
                        "692dcd0882d9ad2168bdffc9ab6a31c9a90dd943, 47ae7e6860dc86f797f2fad6381c698452314dca, "
                        "483cfaabe04025652c1374dde1b952b34df77417, 59cbab196bb2f0006a9dd0fb42d871de7b5a836e, "
                        "fa186a6759d0f9edcebeec21cb92c48e286c74a8, 41f2ead32eaff968e541a65cf23f1d3610b912d7, "
                        "7acd3cfe7e0d9b98a9bdc6c6dda83f7a7046bcdd, 178fb3e90c21f11e22d02777e872a523ce18b92f, "
                        "94f176d98a0c51f0fd9b2b50037c776596765349, 88387f12af2a499b259f8e4632783419475e62c5, "
                        "182b3b57ba864c442d81a0e61076c3d1f2b193d2, 120dfb418b0370e3e603080b5d63311a791d473e"
    }, {
        "id": "8f356c87210201ce44008478caa15320f8cdbb27",
        "title": "Power Control Identification: A Novel Sybil Attack Detection Scheme in VANETs Using RSSI",
        "year": "2019",
        "abstract": "Vehicular ad hoc networks (VANETs) have far-reaching application potentials in the intelligent "
                    "transportation system (ITS) such as traffic management, accident avoidance and in-car "
                    "infotainment. However, security has always been a challenge to VANETs, which may cause "
                    "severe harm to the ITS. Sybil attack is considered as a serious security threat to VANETs "
                    "since the adversary can disseminate false messages with multiple forged identities to "
                    "attack various applications in the ITS. RSSI-based Sybil nodes detection is an efficient "
                    "scheme against Sybil attacks, which adopts position estimation, distribution verification "
                    "or similarity comparison to identify Sybil nodes. However, when Sybil nodes conduct power "
                    "control to deliberately change transmission powers, the received RSSI values would change "
                    "correspondingly, which leads to inaccurate localization or different RSSI time series of "
                    "these Sybil nodes. Thus, it is very difficult to differentiate Sybil nodes from normal nodes "
                    "via conventional RSSI-based methods. This paper first discusses potential power control "
                    "models (PCMs) for launching Sybil attacks in VANETs, then presents two simple Sybil attack "
                    "models and three sophisticated Sybil attack ones with or without power control in detail, "
                    "finally proposes a power control identification Sybil attack detection (PCISAD) scheme to find "
                    "anomalous variations in RSSI time series, which are then used to identify Sybil nodes via a "
                    "linear SVM classifier. Extensive simulations and real-world experiments prove that the proposed "
                    "scheme can effectively deal with Sybil attacks with power control.",
        "outCitations": "e2bf792c3726008ccad56e26400e5a6f7b1c23fa, 8c760d07730ded11fe472e8bb673d4635a92420f, "
                        "b9422aa281460401a5827d95b4467a51101db964, d4b1d4fe1e4dd8e29d3d2fd212d1c10fd92d75d8, "
                        "1b90a06da1a48e83fe1eae66ce4077aa19a48b40, acd13b907c9acf16c6bf7c0a1cff0d7f2402c669, "
                        "547b5562cff99e6fa53c9bd7a53fa342e942822a, efaf77bb979722dabb1cbd05e589ed1dec5899ea, "
                        "e45bb081cd31c24c5f89c6a8a1489671c8bbb109, 5b470a22a6aad4dddd55b2c74b3fd1dce368e17a, "
                        "21c0b29641ab51c505b244701a69b470646f0c70, 85f1d7b48b3be6628ca07fb7767b6c2677d77fd5, "
                        "35516916cd8840566acc05d0226f711bee1b563b, 05b1370358b5d12e6df00d52284620fe2b56a85a, "
                        "def7f97aaad827b62fd6c9e367554b994aea5182, ba32c0ddfe8ebf81f7ce4f8f443912be009e7c41, "
                        "8fa252f6ba387245e8039c4fa0cac0d78e5004a8, 2313d55f4e7739de529e966521018c9043c3fe45, "
                        "cc16f39cf312063dd7f0cb73e9322ebc95d01c9d, c72e5f80ea1e354443768cd1325f143b7723a9da, "
                        "74c9ddb5e5f5d562e6f5372bedf49a4cc7a56e32, da03c8daad3bbbe24de38e8fa39a7f699264e420, "
                        "c5fed112610ece344d915c789c4204afb27c853a, 26d7477c2f05594eb98a191826dcbdb836c9fb45, "
                        "404401e94ef4bc6f401a68decfc7eb834f0a95b5, e9ca41f1c56b2e80bf1b417ceb76b52f2bf59a9b, "
                        "cd62082031ea3488a92bbf00ab308b729a339761, cea967b59209c6be22829699f05b8b1ac4dc092d, "
                        "a3476ba11920d4afa4d0bb9a9b1208daa2c402d2, ad044f5d1d61ea204f35cecd53c86da625e4b2e5, "
                        "9534554af2c5d9988603b4e8da41b9ac81018228, 55fcda7fc495b5fef8d00889e00761938ac9305e"
    }, {
        "id": "e2e52b9fc0fed0b0f26d9b44bc622372030bba74",
        "title": "Modeling the Effects of Transient Populations on Epidemics",
        "year": "2012",
        "abstract": "A large number of transients visit big cities on any given day and they visit crowded areas "
                    "and come in contact with many people. However, epidemiological studies have not paid much "
                    "attention to the role of this subpopulation in disease spread. In the present work, we extend "
                    "a synthetic population model of Washington DC metro area to include leisure and business "
                    "travelers. This approach involves combining Census data, activity surveys, and geospatial data "
                    "to build a detailed minute-byminute simulation of population interaction. We simulate a flu-like "
                    "disease outbreak both with and without the transient population to evaluate the effect of the "
                    "transients on outbreak size and peak day in terms of number of residents infected. Results show "
                    "that there are significantly more infections when transients are considered. We also evaluate "
                    "interventions like closing big museums and encouraging use of hand sanitizers at those museums. "
                    "Surprisingly, closing museums does not result in a significant difference in the epidemic. "
                    "However, we find that if the use of hand sanitizers reduces the infectivity and susceptibility "
                    "to 80% or 60% of the original values, it is as effective as closing museums for a few days or "
                    "entirely eliminating the effect of transients. If infectivity and susceptibility are reduced to "
                    "40% or 20%, it reduces the number of resident infections over the period of 120 days by 10% and "
                    "13%.",
        "outCitations": "c8b80b59ca414dcd5cfa4bb431f722235b9504c0, d5c54719df80fc0de92eb8790ebd4fcd91e6fcc6, "
                        "9c051cc81eb7ea079e12ecdd57564a983debf2e6, feb423ce039a0e82693848c748ca06181e392667, "
                        "1542055ef1d36f656dd00baa26f5a5f77ce750b3, ee3b602bf7ea92550984e84d799d0b1855a0df0b, "
                        "6c73b2f8e15c69822f37fe20cc6413db2234b203, e29498eef434de71947959d5a0209ff14cdd5bfd, "
                        "be3db281ae11ed3ad08c5ac4cf101edf243a8bed, 93cf13d5630806222279db996320d4e8cb54c0b2, "
                        "e50573b554cfa9ee77dcc2e298d7073a152b7199, 9d0c4389436d3ab8ed329dba8c58e8ac6737fd3b, "
                        "33e1d43c3a299d752c18d906c954c6e0fa21e5fe, 1413dbfbbae1b59d52656db3dc48a4ee278e082f, "
                        "54241c2c7caead2f21cebda7722da12061347385, 54fd7f02e1eafa35b0e8f5ff269829c55621bbed"
    }, {
        "id": "868d818208a36d45521cc4bd4e74e3408700d9f5",
        "title": "Photon mapping with visible kernel domains",
        "year": "2018",
        "abstract": "Despite the strong efforts made in the last three decades, lighting simulation systems still "
                    "remain prone to various types of imprecisions. This paper specifically tackles the problem "
                    "of biases due to density estimation used in photon mapping approaches. We study the fundamental "
                    "aspects of density estimation and exhibit the need for handling visibility in the early stage "
                    "of the kernel domain definition. We show that properly managing visibility in the density "
                    "estimation process allows to reduce or to remove biases all at once. In practice, we have "
                    "implemented a 3D product kernel based on a polyhedral domain, with both point-to-point and "
                    "point-to-surface visibility computation. Our experimental results illustrate the enhancements "
                    "produced at every stage of density estimation, for direct photon maps visualization and "
                    "progressive photon mapping.",
        "outCitations": "5bc48081fc7219ab3dc761fcdde27edab21ffd66, 85445352bd3edf8b2bb9dba03b44d3f443f51ea7, "
                        "0706953b0eaca1c6db2160c05429376178172241, 65d4bad6f027397da369364ed5adc8f5bdbcc308, "
                        "c46fd4f725fc6b683efe4c28180f394a4b2a5629, 73da57be973e4d454cf4a416542272198f5ae115, "
                        "c1bb34aea91380004cf24840a3b8eec90dc6eaed, ce8ac018f47144ff91bb6e73e76a1e9c6f322169, "
                        "806dc0fafbd268681cf3dd1090699711c644697d, 6380ce5d03ba0f482420ee21a453ed8bc11cff6b, "
                        "8b382ec13cd638e045c4028a7571dff096352f04, 6d4f6cee4468c55e67e7437bb7579c8c134fe65c, "
                        "b2b73f1a526a5d8064cecc61473c20bec6644942, b211079aac55cd6ddb136e4f37b4f90f2ccefda4, "
                        "c87f5f7653ebfeee00f5ecaddef0f8f6cbdf3733, 28ed3b957243015fd09f54443244cc320844edcb, "
                        "bf03c61f0839b54c51d4e66bc0b030a5dca51d7a, 770f0449a64bc2767d1408464416731057fe9095, "
                        "1b08f4e0c0f70769ce36a23c03e5a1583f3e8064, f2e961943ca3ba671f63ea0f64a011552c9484fa, "
                        "334f6d31fca8c6b8ccfde002d8d81962c616c9ba, 94262431519caf65649453ebd672f2512170d14d, "
                        "7ba546e8371be70c554f5212f9e0c3d006176e91, 4a7d5312eea816179815694c761d215a997c44e3, "
                        "a2d9e138aba892f8573abf1c71e41aa99edbe53c, 5aa52d5c66b04c0bc979ff00a013f722059fa5f1, "
                        "77b64f29aedd529a287e9c6c5aa4200b379677db, 2a889bbe6b12329b4f5bdf9285cd765944be4b11"
    }, {
        "id": "d3ffe71aeca696fc156f9cf7a3d3c292ddd065d1",
        "title": "Automated Differentiation between Normal Sinus Rhythm, Atrial Tachycardia, Atrial Flutter and Atrial "
                 "Fibrillation during Electrophysiology",
        "year": "2017",
        "abstract": "Intracardiac Electrogram (IEGM) are examined during Cardiac Electrophysiology (EP) for detection, "
                    "differentiation, analysis and treatment of different arrhythmias. The arrhythmia detection "
                    "involves EP stimulation, observing IEGM response on monitor screens, and manual evaluation of "
                    "IEGM key features. The process is time consuming and requires high level of expertise of "
                    "Electro physiologists. During an EP stimulation process, a patient may develop Atrial "
                    "Fibrillation (AF) and it is important for patient to be taken out of the AF before further "
                    "proceeding with the procedure. It is required to automate the arrhythmia detection process "
                    "during an EP study for real time monitoring of the patient condition and safety. In our "
                    "previous work, successful detection of Atrio-Ventricular Reentrant Tachycardia and "
                    "Atrio-Ventricular Nodal Re-entry Tachycardia was achieved in time domain. This work has "
                    "been undertaken to automatically detect the AF as well as differentiate it from Atrial Flutter "
                    "(AFL), Atrial Tachycardia (AT) and Normal Sinus Rhythm (NSR). In proposed work, non parametric "
                    "technique has been applied on atrial IEGM signal for estimation of Dominant frequency (DF) to "
                    "find out atrial activation rate during NSR, AT, AFL and AF. A new spectral parameter, Average "
                    "Power Spectral Ratio (APSR), has been defined for ensuring reliability of DF for AF detection "
                    "as well as differentiation of AF from other atrial arrhythmias. The proposed system successfully "
                    "detects and differentiates between NSR, AT, AFL and AF with an accuracy of 99.52%. The proposed "
                    "system can also be effectively used for additional therapeutic application by implantable "
                    "cardioverter defibrillators.",
        "outCitations": "a4227d2c4d6937fccbcd4a352b63da0f73096c5d, c6adc033ceb6cdd21b65c058dd8f82b487bde2ff, "
                        "8b876c793671ba2b0cab2facbf9932e4c05b6672, 950963148f853e0239840ba40bf763e2de24ab37, "
                        "fbec88ea9673d310ed0326c1fcee226ac4139f4a, 8fedab677c0c065f0c8072afc948a4d2255f7d61, "
                        "99d13bf54ae2153c3fad0a05bbc39953b70e524f, 64249d9a493f2a5227ed13c01533d658f47c19e5, "
                        "42d8d63428551a1d8f55d74e3dfab1ee3d4143e7, 2428ca2a328e6ee037b1d238b77ea756119a2193, "
                        "046a84f2f334110f9ff177a6a678674a47f95212, 0bb37a1ee70f73dce4a5a33af5216e3d99ba70de, "
                        "0dfa5e8f9f029d44c95b348bd7d7372ae5b3fcb4, 4bceca0b9e62504b4a5f7d8a8fa5a02ef3071075, "
                        "c11d0ba3e581e691f1bee0022dc0807ff4c428f2, ddd0810fc104ec84c3c63997892ce87bb3dfc9ff, "
                        "9c821eec401ac6a2ebd38fc8d888088b08d88b96, 3fdc2d1044f0379b0ef95403c66aa92b9217f5e1, "
                        "6cdd09f189b1add16a111be38f8321466ee029a4, 9ddeca8756b6d0fc0a19f9b1dad80093ef866f01, "
                        "d3d0bcaf2a10c641233e7b124dc8bcc9c9548836, 1622c1496a8cbba83627b1348706c03b2650dbd2, "
                        "44138929408bde80aad657a8966875e0a16c5489, 0a6a70742f45a97e247d8294b1fbedca7123c0bd, "
                        "4dacf49fda9fd858ac61bc521621b257dbf4f092, fd3ec4f8c062df22c147b4b277d2a0899d7d8462, "
                        "181e7187873184c4156dd5a5e1808a2478e19830, b66a81a259f6106cc93f29733c59a0829daed5ec, "
                        "67c1a61301d3f67f30d8082e5c57e9a2c13931d4"
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
