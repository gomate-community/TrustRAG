from trustrag.modules.vector.embedding import SentenceTransformerEmbedding
from trustrag.modules.engine.weaviate_cli import WeaviateEngine
if __name__ == '__main__':
    # 初始化 MilvusEngine
    local_embedding_generator = SentenceTransformerEmbedding(model_name_or_path=r"H:\pretrained_models\mteb\all-MiniLM-L6-v2", device="cuda")
    weaviate_engine = WeaviateEngine(
        collection_name="startups",
        embedding_generator=local_embedding_generator,
        client_params={
            "http_host": "localhost",
            "http_port": 8080,
            "http_secure": False,
            "grpc_host": "localhost",
            "grpc_port": 50051,
            "grpc_secure": False,
        }
    )

    documents = [
        {"name": "SaferCodes", "images": "https://safer.codes/img/brand/logo-icon.png",
         "alt": "SaferCodes Logo QR codes generator system forms for COVID-19",
         "description": "QR codes systems for COVID-19.\nSimple tools for bars, restaurants, offices, and other small proximity businesses.",
         "link": "https://safer.codes", "city": "Chicago"},
        {"name": "Human Practice",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/373036-94d1e190f12f2c919c3566ecaecbda68-thumb_jpg.jpg?buster=1396498835",
         "alt": "Human Practice -  health care information technology",
         "description": "Point-of-care word of mouth\nPreferral is a mobile platform that channels physicians\u2019 interest in networking with their peers to build referrals within a hospital system.\nHospitals are in a race to employ physicians, even though they lose billions each year ($40B in 2014) on employment. Why ...",
         "link": "http://humanpractice.com", "city": "Chicago"},
        {"name": "StyleSeek",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/3747-bb0338d641617b54f5234a1d3bfc6fd0-thumb_jpg.jpg?buster=1329158692",
         "alt": "StyleSeek -  e-commerce fashion mass customization online shopping",
         "description": "Personalized e-commerce for lifestyle products\nStyleSeek is a personalized e-commerce site for lifestyle products.\nIt works across the style spectrum by enabling users (both men and women) to create and refine their unique StyleDNA.\nStyleSeek also promotes new products via its email newsletter, 100% personalized ...",
         "link": "http://styleseek.com", "city": "Chicago"},
        {"name": "Scout",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/190790-dbe27fe8cda0614d644431f853b64e8f-thumb_jpg.jpg?buster=1389652078",
         "alt": "Scout -  security consumer electronics internet of things",
         "description": "Hassle-free Home Security\nScout is a self-installed, wireless home security system. We've created a more open, affordable and modern system than what is available on the market today. With month-to-month contracts and portable devices, Scout is a renter-friendly solution for the other ...",
         "link": "http://www.scoutalarm.com", "city": "Chicago"},
        {"name": "Invitation codes", "images": "https://invitation.codes/img/inv-brand-fb3.png",
         "alt": "Invitation App - Share referral codes community ",
         "description": "The referral community\nInvitation App is a social network where people post their referral codes and collect rewards on autopilot.",
         "link": "https://invitation.codes", "city": "Chicago"},
        {"name": "Hyde Park Angels",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/61114-35cd9d9689b70b4dc1d0b3c5f11c26e7-thumb_jpg.jpg?buster=1427395222",
         "alt": "Hyde Park Angels - ",
         "description": "Hyde Park Angels is the largest and most active angel group in the Midwest. With a membership of over 100 successful entrepreneurs, executives, and venture capitalists, the organization prides itself on providing critical strategic expertise to entrepreneurs and ...",
         "link": "http://hydeparkangels.com", "city": "Chicago"},
        {"name": "GiveForward",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/1374-e472ccec267bef9432a459784455c133-thumb_jpg.jpg?buster=1397666635",
         "alt": "GiveForward -  health care startups crowdfunding",
         "description": "Crowdfunding for medical and life events\nGiveForward lets anyone to create a free fundraising page for a friend or loved one's uncovered medical bills, memorial fund, adoptions or any other life events in five minutes or less. Millions of families have used GiveForward to raise more than $165M to let ...",
         "link": "http://giveforward.com", "city": "Chicago"},
        {"name": "MentorMob",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/19374-3b63fcf38efde624dd79c5cbd96161db-thumb_jpg.jpg?buster=1315734490",
         "alt": "MentorMob -  digital media education ventures for good crowdsourcing",
         "description": "Google of Learning, indexed by experts\nProblem: Google doesn't index for learning. Nearly 1 billion Google searches are done for \"how to\" learn various topics every month, from photography to entrepreneurship, forcing learners to waste their time sifting through the millions of results.\nMentorMob is ...",
         "link": "http://www.mentormob.com", "city": "Chicago"},
        {"name": "The Boeing Company",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/49394-df6be7a1eca80e8e73cc6699fee4f772-thumb_jpg.jpg?buster=1406172049",
         "alt": "The Boeing Company -  manufacturing transportation", "description": "",
         "link": "http://www.boeing.com", "city": "Berlin"},
        {"name": "NowBoarding \u2708\ufe0f",
         "images": "https://static.above.flights/img/lowcost/envelope_blue.png",
         "alt": "Lowcost Email cheap flights alerts",
         "description": "Invite-only mailing list.\n\nWe search the best weekend and long-haul flight deals\nso you can book before everyone else.",
         "link": "https://nowboarding.club/", "city": "Berlin"},
        {"name": "Rocketmiles",
         "images": "https://d1qb2nb5cznatu.cloudfront.net/startups/i/158571-e53ddffe9fb3ed5e57080db7134117d0-thumb_jpg.jpg?buster=1361371304",
         "alt": "Rocketmiles -  e-commerce online travel loyalty programs hotels",
         "description": "Fueling more vacations\nWe enable our customers to travel more, travel better and travel further. 20M+ consumers stock away miles & points to satisfy their wanderlust.\nFlying around or using credit cards are the only good ways to fill the stockpile today. We've built the third way. Customers ...",
         "link": "http://www.Rocketmiles.com", "city": "Berlin"}

    ]
    vectors = weaviate_engine.embedding_generator.generate_embeddings([doc["description"] for doc in documents])
    print(vectors.shape)
    payload = [doc for doc in documents]

    # Upload vectors and payload
    weaviate_engine.upload_vectors(vectors=vectors, payload=payload)
    


    # 构建过滤器并搜索
    conditions = [
        {"key": "city", "match": "Berlin"},
    ]
    custom_filter = weaviate_engine.build_filter(conditions)

    # 搜索柏林的度假相关创业公司
    results = weaviate_engine.search(
        text="vacations",
        # query_filter=custom_filter,
        limit=5
    )
    print(results)
    for result in results:
        print(result)