import requests
import json

# 请替换为您的访问令牌
ACCESS_TOKEN = 'your_facebook_access_token'


def get_post_comments(post_id, access_token):
    # 构造 API 请求 URL，指定要检索的数据字段
    url = f"https://graph.facebook.com/v13.0/{post_id}/comments"
    params = {
        "access_token": access_token,
        "fields": "message,from,created_time",  # 指定需要的字段
        "limit": 100  # 每次请求最多获取 100 条评论
    }

    comments = []
    while url:
        # 发起 GET 请求
        response = requests.get(url, params=params)
        data = response.json()

        if 'data' in data:
            comments.extend(data['data'])

        # 检查是否有下一页
        url = data.get('paging', {}).get('next', None)

    return comments


# 获取某个主题下的帖子 ID 列表（可以通过特定页面的帖子 ID 直接获取）
def get_posts_for_topic(page_id, access_token, topic):
    url = f"https://graph.facebook.com/v13.0/{page_id}/posts"
    params = {
        "access_token": access_token,
        "fields": "message,id",  # 获取帖子内容和 ID
        "limit": 10  # 可以根据需要设置更高的限制
    }

    posts = []
    response = requests.get(url, params=params)
    data = response.json()

    for post in data['data']:
        if topic.lower() in post.get("message", "").lower():
            posts.append(post['id'])

    return posts


# 示例调用
page_id = 'your_target_page_id'  # 替换为目标页面 ID
topic = "your_topic"  # 替换为要搜索的主题
post_ids = get_posts_for_topic(page_id, ACCESS_TOKEN, topic)

all_comments = {}
for post_id in post_ids:
    comments = get_post_comments(post_id, ACCESS_TOKEN)
    all_comments[post_id] = comments

# 将评论保存到文件
with open("comments.json", "w", encoding="utf-8") as f:
    json.dump(all_comments, f, ensure_ascii=False, indent=4)

print("评论已保存到 comments.json 文件中。")
