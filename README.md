# Painting style transfer
**Authors - Zdenek Jelinek, Adam Gregor**

**Year - 2021**

# Link to repository
`https://github.com/Zjelin/paint_style_transfer`

# Install
`pip3 install -r requirements.txt`

# Run
`python3 style_transfer -c c_path -s s_path [-o optim|-l layers|-h]`

-c - Path to content image, e.g. `data/content/train.jpg`

-s - Path to style image, e.g. `data/style/composition.jpg`

-o - Select optimizer, options are `adam` or `lbfgs`

-l - Select layers, options are `original` or `alternative`