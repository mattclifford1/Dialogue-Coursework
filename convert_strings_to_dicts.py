import pandas as pd
import ast

# converters
to_dict = lambda ex:ast.literal_eval(ex)
df1 = pd.read_csv('./DNCoursework/data/doc2dial_rc_val.csv', converters={'answers':to_dict, 'spans':to_dict})

# df1 = pd.read_csv('./DNCoursework/data/doc2dial_rc_train.csv')
print(df1.columns)
print(type(df1["answers"].iloc[0]))

# df2 = df1.copy()
# df2 = df2.drop(['Unnamed: 0', 'answers', 'spans'], axis=1)


# df2['answers'] = df1['answers'].apply(ast.literal_eval)
# df2['spans'] = df1['spans'].apply(ast.literal_eval)

# df2.to_csv('doc2dial_rc_train.csv', index=False)
# print(df2.columns)
# print(type(df2["answers"].iloc[0]))

# df3 = df1[['id', 'question', 'context', 'spans', 'answers', 'domain', 'title']]
# df3.to_csv('doc2dial_rc_train.csv', index=False)

df1.to_csv('doc2dial_rc_val.csv.zip', index=False, compression="zip")
