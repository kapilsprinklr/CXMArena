def get_kb_content(articles_df, kb_id):
    rows = articles_df.loc[articles_df.document_id == kb_id]
    return rows.document_content.tolist()[0]


def get_kbs_content_string(articles_df, kb_ids):
    if not kb_ids:
        return "NONE"

    all_kbs = [f"KB {i + 1} ID: {kb_id}\n KB {i + 1} Content: {get_kb_content(articles_df, kb_id)}" for i, kb_id in
               enumerate(kb_ids)]
    line_br = '-' * 50
    return f"\n{line_br}\n".join(all_kbs)