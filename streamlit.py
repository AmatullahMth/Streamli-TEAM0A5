import streamlit as st

def main():
    # collect user text
    news_text = st.text_area("Enter News Text: ", height = 200)

    model = st.selectbox(
    "How would you like to be contacted?",
    ("MultinomialNB", "SVC", "Logistic Regression"))

    st.write("You selected:", model)

    
    # Create the classifier
    if st.button("Classify"):
        if news_text:
            # convert user text into a df
            userdf = pd.DataFrame({'all_text': [news_text]})
        
            # perform preprocessing using our preprocessing function - df_processor
            cleandf = df_processor(userdf)
        
            # convert to features using my vectorizer
            X_ft = vectorizer.transform(cleandf['all_text'])
        
            # perform a prediction on the vectorised text
            if model == "MultinomialNB":
                y_pred = modelNB.predict(X_ft)
            elif model == "SVC":
                y_pred = modelSVC.predict(X_ft)
            elif model == "Logistic Regression":
                y_pred = modelLR.predict(X_ft)
        
            # inverse transform and print the category
            y_tran = le.inverse_transform(y_pred)
            readable_cat = y_tran[0].title() 
        
            # output to the reader
            st.success(readable_cat)
        else:
            st.warning("Please enter some news text.")



if __name__ == "__main__":
    main()

