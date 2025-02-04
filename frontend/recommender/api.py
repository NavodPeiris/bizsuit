from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import generate_recommendations

app = FastAPI()

# Define response model
class ProductResponse(BaseModel):
    prod_id: str
    category_code: str

# Endpoint to get product details by product_id
@app.get("/get_recommendation/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str):
    prod_id, category_code = generate_recommendations(product_id)
    print("prod_id: ", prod_id)
    print("category_code: ", category_code)
    if(prod_id is None or category_code is None):
        return HTTPException(status_code=404, detail="Recommendation not found")
    return ProductResponse(prod_id=prod_id, category_code=category_code)