package com.example.refresh_selection;

public class BottomCard {
    private String bottom_space_name;
    private String bottom_space_description;
    private String distance;
    private String price;
    private String img;
    private String mlsfc;
    private String mcate;

    BottomCard(String btm_space_name,String btm_space_description,String dis,String price, String img,String mlsfc, String mcate){
        this.bottom_space_name=btm_space_name;
        this.bottom_space_description=btm_space_description;
        this.distance=dis;
        this.price=price;
        this.img=img;
        this.mlsfc=mlsfc;
        this.mcate=mcate;
    }

    public String getBottom_space_name() {
        return bottom_space_name;
    }

    public void setBottom_space_name(String bottom_space_name) {
        this.bottom_space_name = bottom_space_name;
    }

    public String getBottom_space_description() {
        return bottom_space_description;
    }

    public void setBottom_space_description(String bottom_space_description) {
        this.bottom_space_description = bottom_space_description;
    }

    public String getDistance() {
        return distance;
    }

    public void setDistance(String distance) {
        this.distance = distance;
    }

    public String getPrice() {
        return price;
    }

    public void setPrice(String price) {
        this.price = price;
    }

    public String getImg() {
        return img;
    }

    public void setImg(String img) {
        this.img = img;
    }

    public String getMlsfc() {
        return mlsfc;
    }

    public void setMlsfc(String mlsfc) {
        this.mlsfc = mlsfc;
    }

    public String getMcate() {
        return mcate;
    }

    public void setMcate(String mcate) {
        this.mcate = mcate;
    }
}
