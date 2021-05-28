def print_example(data, indices):
    selected_rows = data.loc[indices]
    for idx, row in selected_rows.iterrows():
        cmd = input()
        if cmd == "s":
            break
        print( "%s | %s >> " %  (row['Title'], row["Label"]))
        cmd = input()
        if cmd == "a":
            if type(row['Abstract']) == str:
                abstract = row['Abstract']
                #abstract = abstract.replace('. ', '.\n')
                print(abstract)
        else:
            continue
