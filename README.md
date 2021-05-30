# Painting style transfer
**Authors - Zdenek Jelinek, Adam Gregor**

**Year - 2021**

# Install
`pip3 install -r requirements.txt`

# Run
`python3 style_transfer -c c_path -s s_path [-o optim|-l layers|-h]`

-c - Path to content image

-s - Path to style image

-o - Select optimizer, options are `adam` or `lbfgs`

-l - Select layers, options are `original` or `alternative`