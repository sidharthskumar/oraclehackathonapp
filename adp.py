from flask import Flask, request, Response
import requests
import json
import asyncio
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import time
from datetime import datetime
from langchain_community.llms import OCIGenAI
import oci


print(PromptTemplate)


app = Flask(__name__)

@app.route("/apis/langchain", methods=["POST"])
def langchainflow():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("vectorstores", embeddings)
    rel = db.similarity_search("dogs", k=1)
    info = rel[0].page_content

    prompt=link=""
    request_json=request.get_json()
    if "prompt" in request_json:
        prompt = request_json["prompt"] 
    if "link" in request_json:
        link = request_json["link"]     
    if link!="":
        prompt == "new requirement=prompt+pagecontent" #append promopt and extracted content from confluence

        PromptTemplatewithlink = f"""
    Using the contextual information delimited by <info> </info> and the requirement delimited by <req> </req> provide the response in format delimited by <example> </example>

    <info>
    {info}
    </info>
    <req>
    {prompt}
    </req>

    <example>
    {{
        "ApplicationName": A professional short and meaningful name for the application ,
        "Tables": Array of tables required to build the application as string array
        "SQLCommands": Array of create tables SQL command with 20 most relevant columns to create the tables in Oracle database as string array with each sql command as an item
        "Pages": Object Array of different pages reqired to build the application, object having keys
        {{
            "PageName" : Name for the Page,
            "PageType": whether the page is list_page , details_page, list_details_page, dashboard_page, calendar_page,
            "TableAssociated" : Table associated with the page,
            "FieldsRequired" : The title field in the table
        }}
    }}
    </example>
    """
    if link=="":
        PromptTemplate = f"""
    Using the contextual information delimited by <info> </info> and the requirement delimited by <req> </req> provide the response in format delimited by <example> </example>

    <info>
    {info}
    </info>
    <req>
    {prompt}
    </req>

    <example>
    {{
        "ApplicationName": A professional short and meaningful name for the application ,
        "Tables": Array of tables required to build the application as string array
        "SQLCommands": Array of create tables SQL command with 20 most relevant columns to create the tables in Oracle database as string array with each sql command as an item
        "Pages": Object Array of different pages reqired to build the application, object having keys
        {{
            "PageName" : Name for the Page,
            "PageType": whether the page is list_page , details_page, list_details_page, dashboard_page, calendar_page,
            "TableAssociated" : Table associated with the page,
            "FieldsRequired" : The title field in the table
        }}
    }}
    </example>
    """

    llm = OCIGenAI(
    model_id="cohere.command",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id="ocid1.tenancy.oc1..aaaaaaaad7nju4v2lioxocqnkeqydccehtrswmbsievlnhie2rrqguu5ruxq",
    model_kwargs={"temperature": 0.7, "max_tokens": 4000},
    )
    response = llm.invoke(PromptTemplate, temperature=0.7)
    print((response))
    response = response[int(response.find("{")):int(response.rfind("}"))+1]
    response = response.replace("\n","")
    response = response.replace("\\","")
    
    print((response))
    return {"result":json.loads(response)}



@app.route("/apis/updateVector", methods=["POST"])
def updatevectordb():
    key_content = '-----BEGIN RSA PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDAD9p6H0FRMvFa bOnrH+SVi55CeE1Y1gwAxUSnvJoDH9HAfHIseHdrqcIc5NqxfAneQb3tbH58ZZlj ZyL+s1cXCk4kD/bNMDhPaoXKFxziTqMEnYHboDqOYKQtTaL15jkrjVCA8RKhfD07 kBcRqyrM0bLfALVRpNiPWzHH0MLavazx/SqHQaqSeHLtZQAYKi5gdM2xe5fhLMPd aDr/g0sAYbZzWGlMjNjvqlcHM7kt1Z1IxmLSGjIpwnkkzx2WNMRZ/eInyM4OSkpb psCQsZOyVYVIxk0KUy+AI2U0gxGQzZWYXNn+fQaftl88OmeuC7R6n1gIoyz0Q8CM Sh4V0codAgMBAAECggEACHYqV/Wx46lCPOD42qUIZcDwSK69fvZ0h5bT/tfY/ojp As6jcRYgR3FZM1CvLzaZxghQx88J240rrtdyZpTxd1BllO0nIG8ewVDzcq3bVepv qtpi8ubM7aI7BCFG1rKSk4/28KyuTMhed5X844hjqwEUx60OT/namKZEkTBqBUer qaEY359fgzUxVw6CBRRTTR9jnB+Lsk2/TYPb5wbehgEx5tMqgTGbJ572EtnvMAmT RPgSBOubls6EzNp3F5CM/arAkIHRt/rvcA/lHBniYLsqq+rZ81Cy3Mxg7gYE94xk uLZo/XI7WPtvHQ5xxr3zG7TTNG1Blex1lE1m/TYZAQKBgQD5fHf3815oB8ceDeng LzGnlR3VuofLeH8KkpE2yaATT44Fy4RqMWku2q7xHkagYALEB0qtU8v8gNX+xAwi buN04DZ8FyOsf/uDeJ8hO1frnuf3hN1vt6DECS0vJ+WdbDxa+GH/h87rnBj6LopX wlS1VH1Ra2BBtdFTVGBu3y6d4QKBgQDFE5Gf+7hOMZzFsIDDQ5zu7woP8mquoaZP nLiCRSDZRpMIjGzaIZKFgMk96OzqhjwBZsfA32UoYfvSclKt5VMnFcooAeSYBxFq HapsFiJJzU5lti35cavm7akKCWXr/p34+apux7pjjRdyIdLr78Iud2nSQE0z5vZI gOJb0NCbvQKBgQD5Zg3fli/nuu64ApyreUIgDpb2kzMwmdIV1ZLIvCIDa+HDtUE3 jxFgv0dmzic4JwJcyBVE06H1Vy2VMpIW0dcbfQ+6WL9Wr9HUCX66D8LCTeYBr5ZV GbHihnHe0/lbt1lWbzo34aFeMntdjazKMf/QDRgBjl95ELKipJSKAR1uwQKBgHKf 7SVmHU6toIeDH4FzBAYc1ndsAgzMTJUljFOIrZBycfaY5n8A493uiB4QKixGIwSV qT1PMeEJDJTclaY7KeAj1k7quvCJu+FCm+r9/Ld8SEr0aU0ahmdsd9M0oClhELTN UgnY9VoNqENj1PARpZmtLslxSPYVMc392Kqai5rpAoGARIA3UqKzoqYpH2YW3SnT gCkc73pSZy5S+JoXU9j7fiSCmN7boZolJV783T6GpaWY/MKQpH4YOzuuCy/tqxt6 gh7zEwentgTVcdUs9lQ861k6BqmoVlKfJTlE1wnp1utSrqVrQ6oGJOnJYVd2Ooao geGW4h24/TqdVjOv0ndrVK0=\n-----END RSA PRIVATE KEY-----'
    config_with_key_content = {
    "user": 'ocid1.user.oc1..aaaaaaaaz7xwt7pdzh4jqdoedqickgcgacfxu6bkgecv2ubqxew3i4b2ffsa',
    "key_content": key_content,
    "fingerprint": '6c:d2:83:b3:4b:91:84:5c:b0:7d:33:3e:dc:19:56:e4',
    "tenancy": 'ocid1.tenancy.oc1..aaaaaaaad7nju4v2lioxocqnkeqydccehtrswmbsievlnhie2rrqguu5ruxq',
    "region": 'us-ashburn-1'}
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client = oci.object_storage.ObjectStorageClient(config_with_key_content)
    f= client.get_object(namespace_name="cxcomms",bucket_name="oci-pdfs",object_name="fiass.pkl").data.content
    db= FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=f)
    db.save_local("vectorstores")
    return {"result":"saved vectordb to local"}


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9029,debug=True)

