_INDEX = """
<!DOCTYPE HTML>
<head>
  <style type="text/css">
html, body, #image {
    width: 100%;
    height: 100%;
}
body {
    background: radial-gradient(
        circle at center,
        #000 0%,
        #002 50%,
        #004 65%,
        #408 75%,
        #824 85%,
        #f40 90%,
        #fb0 95%,
        white 100%
    );
}
  </style>
</head>
<body>
  <img id="image" src="/aurora.png"></img>
</body>
</html>
"""


def get_index():
    return _INDEX
