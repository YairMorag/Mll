import streamlit as st
import pandas as pd
import joblib
import numpy as np

def load_model():
    """Load the saved model and its components"""
    try:
        model_data = joblib.load('loan_risk_model.pkl')
        return model_data['model'], model_data['scaler'], model_data['categorical_mappings']
    except Exception as e:
        st.error(f"שגיאה בטעינת המודל: {str(e)}")
        st.stop()

# Configure page
st.set_page_config(page_title="מערכת חיזוי סיכון הלוואות", layout="wide")

# Load model
model, scaler, categorical_mappings = load_model()

# כותרת ראשית
st.title("מערכת חיזוי סיכון הלוואות")

# הסבר על המערכת
st.info("""
### 🎯 על המערכת
מערכת זו מחשבת ציון סיכון להלוואה בסקאלה של 1-10:
* 1-3: סיכון נמוך 🟢
* 4-6: סיכון בינוני 🟡
* 7-10: סיכון גבוה 🔴

ככל שהציון גבוה יותר, כך רמת הסיכון גבוהה יותר.
""")

# יצירת שני טורים ראשיים
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 הזנת נתונים")
    
    # יצירת שתי שורות של שלושה שדות כל אחת
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row1_col1:
        age = st.number_input("גיל:", min_value=18, max_value=100, value=30, step=1)
    
    with row1_col2:
        debt_amount = st.number_input("סכום חוב (₪):", min_value=0.0, value=10000.0, step=1000.0)
    
    with row1_col3:
        family_income = st.number_input("הכנסה משפחתית שנתית (₪):", 
                                      min_value=0.0, value=120000.0, step=1000.0)

    with row2_col1:
        payment_history = st.selectbox(
            "היסטוריית תשלומים:",
            ['excellent', 'good', 'fair', 'poor', 'unknown'],
            format_func=lambda x: {
                'excellent': 'מצוין',
                'good': 'טוב',
                'fair': 'סביר',
                'poor': 'חלש',
                'unknown': 'לא ידוע'
            }[x]
        )
    
    with row2_col2:
        profession = st.selectbox(
            "מקצוע:",
            ['teacher', 'doctor', 'engineer', 'other'],
            format_func=lambda x: {
                'teacher': 'מורה',
                'doctor': 'רופא',
                'engineer': 'מהנדס',
                'other': 'אחר'
            }[x]
        )
    
    with row2_col3:
        num_children = st.number_input("מספר ילדים:", min_value=0, value=0, step=1)

with col2:
    st.markdown("### 📊 יחס חוב להכנסה")
    debt_ratio = debt_amount / family_income if family_income > 0 else 0
    st.progress(min(debt_ratio, 1.0))
    st.write(f"יחס חוב להכנסה: {debt_ratio:.1%}")
    
    if debt_ratio > 0.6:
        st.warning("⚠️ יחס חוב להכנסה גבוה")
    elif debt_ratio > 0.4:
        st.warning("⚠️ יחס חוב להכנסה בינוני-גבוה")

def get_risk_category(score):
    """המרת ציון מספרי לקטגוריית סיכון וצבע"""
    if score <= 3:
        return "סיכון נמוך", "green"
    elif score <= 6:
        return "סיכון בינוני", "orange"
    else:
        return "סיכון גבוה", "red"

# כפתור לחישוב
if st.button("חשב סיכון", type="primary", use_container_width=True):
    try:
        # יצירת DataFrame עם שורה אחת של הנתונים
        input_data = pd.DataFrame([[
            age,
            debt_amount,
            payment_history,
            family_income,
            profession,
            num_children
        ]], columns=['Age', 'Debt amount', 'Payment history', 
                    'Family income', 'Profession', 'Num. of children'])

        # המרת המשתנים הקטגוריים
        input_data['Payment history'] = input_data['Payment history'].map(categorical_mappings['Payment history'])
        input_data['Profession'] = input_data['Profession'].map(categorical_mappings['Profession'])

        # סילום המשתנים המספריים
        numerical_cols = ['Age', 'Debt amount', 'Family income']
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

        # חיזוי
        risk_score = model.predict(input_data)[0]
        risk_category, risk_color = get_risk_category(risk_score)

        # הצגת התוצאות
        st.markdown("---")
        
        # יצירת שלושה טורים להצגת התוצאות
        res_col1, res_col2, res_col3 = st.columns([1,1,2])
        
        with res_col1:
            st.metric(label="ציון סיכון", value=f"{risk_score:.0f}/10")
            
        with res_col2:
            st.markdown(f"""
            ### רמת סיכון
            <div style='padding: 10px; background-color: {risk_color}; border-radius: 5px; color: white; text-align: center;'>
                <strong>{risk_category}</strong>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col3:
            st.markdown("### ניתוח התוצאה")
            st.write(f"""
            * היחס בין החוב להכנסה השנתית: **{debt_ratio:.1%}**
            * רמת הסיכון הכוללת: **{risk_category}**
            * הציון מתבסס על ניתוח של כל הפרמטרים שהוזנו
            """)

            # הצגת המלצות בהתאם לרמת הסיכון
            if risk_score <= 3:
                st.success("✅ פרופיל סיכון טוב - מומלץ להמשיך בתהליך")
            elif risk_score <= 6:
                st.warning("⚠️ מומלץ לבחון דרכים להקטנת רמת הסיכון")
            else:
                st.error("❌ רמת סיכון גבוהה - יש לשקול מחדש את תנאי ההלוואה")

    except Exception as e:
        st.error(f"אירעה שגיאה בעת החישוב: {str(e)}")
        st.write("אנא ודא שכל השדות מולאו כראוי ונסה שוב.")

# מידע נוסף בתחתית העמוד
with st.expander("ℹ️ מידע נוסף על חישוב הסיכון"):
    st.markdown("""
    ### כיצד מחושב ציון הסיכון?
    המערכת משקללת מספר פרמטרים מרכזיים:

    1. **יחס חוב להכנסה**
       * מתחת ל-20%: סיכון נמוך
       * 20%-40%: סיכון בינוני
       * מעל 40%: סיכון גבוה

    2. **היסטוריית תשלומים**
       * מצוין: מפחית את רמת הסיכון
       * חלש: מעלה משמעותית את רמת הסיכון

    3. **גיל**
       * גילאי 25-60: סיכון רגיל
       * מתחת ל-25 או מעל 60: תוספת קטנה לסיכון

    4. **מספר ילדים**
       * משפיע על ההוצאות המשפחתיות

    ### סקאלת הציונים (1-10)
    * **1-3**: סיכון נמוך - סבירות גבוהה להחזר תקין
    * **4-6**: סיכון בינוני - נדרשת בחינה נוספת
    * **7-10**: סיכון גבוה - סבירות נמוכה להחזר תקין
    """)