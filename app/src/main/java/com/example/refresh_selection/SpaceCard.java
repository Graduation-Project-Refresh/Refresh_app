package com.example.refresh_selection;

public class SpaceCard {//카드 클래스
    private String space_name;
    private int rating;
    private String img;
    private int image;//int가 맞나? 경로면 String?
    private String space_description;
    private String space_description2;
    private String mlsfc;
    private String mcate;

    // Constructor
    public SpaceCard(String space_name, String space_description,String space_description2, String image,String mlsfc,String mcate) {
        this.space_name = space_name;
        this.space_description=space_description;
        this.space_description2=space_description2;
//        this.rating = rating;
        this.img = image;
        this.mlsfc= mlsfc;
        this.mcate=mcate;
    }

    // Getter and Setter
    public String getSpace_name() {
        return space_name;
    }

    public void setSpace_name(String course_name) {
        this.space_name = course_name;
    }

    public int getRating() {
        return rating;
    }

    public void setRating(int course_rating) {
        this.rating = course_rating;
    }

    public String getImage() {
        return img;
    }

    public void setImage(String image) {
        this.img = image;
    }

    public String getDescription() {
        return space_description;
    }
    public void setDescription(String space_description){
        this.space_description=space_description;
    }
    public String getDescription2() {
        return space_description2;
    }
    public void setDescription2(String space_description2){
        this.space_description2=space_description2;
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
