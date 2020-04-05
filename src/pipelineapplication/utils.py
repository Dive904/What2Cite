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
