package com.example.refresh_selection;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonArrayRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.bottomsheet.BottomSheetDialogFragment;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Random;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class ModalBottomSheet extends BottomSheetDialogFragment {
    String TAG = "bottomSheet";
    private BottomSheetListener mListener;
    private View view;
    //바텀 시트 안에 카드뷰
    BottomCardAdapter bottomCardAdapter;
    private RecyclerView bottomRV;
    private ArrayList<BottomCard> bottomCardArrayList= new ArrayList<BottomCard>();//카드 정보 저장
    private int Radius =100;

    long seed = System.currentTimeMillis();
    Random rand = new Random(seed);

    public ModalBottomSheet(){//empty constructor

    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
//        mListener = (BottomSheetListener) getContext();
        view = inflater.inflate(R.layout.bottom_sheet, container, false);

        GpsTracker gpsTracker = new GpsTracker(getActivity());
        float lati = (float)gpsTracker.getLatitude(); // 현재 위치 위도
        float longi = (float)gpsTracker.getLongitude();// 현재위치 경도

        requestLatitudeLongtitude(lati,longi,Radius);


        return view;
    }

    //장소 카드 만듣
    public void makeMapSpaceCard(View mainview) {
        bottomRV = (RecyclerView) mainview.findViewById(R.id.bottom_sheet_recycleview);//activitiy_map : bottomsheet layout
        bottomRV.setHasFixedSize(true);
//        bottomCardArrayList.add(new BottomCard("대청호1","호수가 아름답다.","300m","1000원",R.drawable.test_img));
        Log.d("size",bottomCardArrayList.size()+"");
        bottomCardAdapter =new BottomCardAdapter(mainview.getContext(),bottomCardArrayList);//content가 문제인거 같은데..
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        bottomRV.setLayoutManager(linearLayoutManager);
        bottomRV.setAdapter(bottomCardAdapter);
    }
    private void requestLatitudeLongtitude( float latitude, float longitude, int radius) {//서버에 장소 좌표 요청하는 메소드
        RequestQueue queue = Volley.newRequestQueue(getContext());
        String url = String.format("http://3.19.218.150:8089//api/getPlaces?meters=%1$s&x=%2$s&y=%3$s",radius,longitude,latitude);//서버url
        JsonArrayRequest request = new JsonArrayRequest(url ,new Response.Listener<JSONArray>() {
            @Override
            public void onResponse(JSONArray response) {
                Log.d("bottom_JSONArray",response.toString());
                try {
                    for(int i=0; i<response.length();i++){
                        JSONObject place =(JSONObject)response.get(i);
                        String name = place.getString("poi_nm");
                        String descript=place.get("bemd_nm")+" "+place.getString("beonji")+" "+place.getString("branch_nm")+" "+place.getString("mlsfc");

                        String dis=place.getString("mlsfc");//mcate_name

                        String price=(10000+Math.round(rand.nextInt(10000)/100.0)*100)+"";
                        String tag = place.getString("id");
//                        String img= place.getString("imgSrc");
                        String photoUrl="http%3A%2F%2Fwww.badastar.or.kr%2Fstatic%2Fuser%2Fimages%2Fmain%2Fs1_slide3.jpg&imgrefurl=http%3A%2F%2Fwww.badastar.or.kr%2F&tbnid=I-28HQ5b6cd5bM&vet=12ahUKEwitnJXFj4P0AhWrxYsBHYjaCmcQMygDegUIARDNAQ..i&docid=3OSUQx31TCsYeM&w=1920&h=1081&q=%EB%B0%94%EB%8B%A4&ved=2ahUKEwitnJXFj4P0AhWrxYsBHYjaCmcQMygDegUIARDNAQ";
//                        String photoUrl="http://18.218.2.153/"+img;

//                        String mlsfc=place.getString("mlsfc");
//                        String mcate=place.getString("mcate_nm");

//                        BottomCard cd=new BottomCard(name,descript,dis,price,photoUrl,mlsfc,mcate);
//                        bottomCardArrayList.add(cd);//바텀 시트에 카드뷰생성
                    }
//                    Tag = jsonResponse.getInt("tag");//마커 구분 번호?
                } catch (JSONException e) {
                    e.printStackTrace();
                }
                bottomCardArrayList.add(new BottomCard("아웃닭","치킨집","식당","16000","https://t1.daumcdn.net/cfile/tistory/265CAD3B58958AED16","식당","치킨"));
                bottomCardArrayList.add(new BottomCard("카페누오보","커피집","카페","6500","https://mp-seoul-image-production-s3.mangoplate.com/47875_1495081318224150.jpg","식당","카페(도심)"));
                bottomCardArrayList.add(new BottomCard("맘스터치","햄버거","식당","6000","https://t1.daumcdn.net/cfile/tistory/99D0BD4E5E9FE59904","식당","햄버거/도너츠/샌드위치"));
                bottomCardArrayList.add(new BottomCard("어썸","커피집","카페","5000","https://mblogthumb-phinf.pstatic.net/20160421_64/yunah0218_1461166491572WQ9pq_JPEG/KakaoTalk_Photo_2016-04-21-00-28-09_78.jpeg?type=w800","식당","카페(도심)"));
                bottomCardArrayList.add(new BottomCard("라드커피","커피집","카페","5000","https://mblogthumb-phinf.pstatic.net/MjAxODA3MDhfMjMg/MDAxNTMxMDIwMzIyNDcy.rmua8885oxPr7nhBYrn_m9LdFPQ5PhINXjnv_troE3gg.ctlysxaXhF-n8zqzLRH9_1NIJoUHkZkqLb5-SwP375wg.JPEG.ppo1127/image_5941379981531020290018.jpg?type=w800","식당","카페(도심)"));
                bottomCardArrayList.add(new BottomCard("아웃닭","치킨집","식당","16000","https://t1.daumcdn.net/cfile/tistory/265CAD3B58958AED16","식당","치킨"));

                makeMapSpaceCard(view);
            }
//
//            @Override
//            public void onResponse(JSONObject response) {
//                Log.d("JSONArray",response.toString());
        }, new Response.ErrorListener() {

            @Override
            public void onErrorResponse(VolleyError error) {
                Log.d("responseError", "위도, 경도 가져오기 실패 "+error.getMessage());
            }
        });
        request.setRetryPolicy(new DefaultRetryPolicy(DefaultRetryPolicy.DEFAULT_TIMEOUT_MS, DefaultRetryPolicy.DEFAULT_MAX_RETRIES, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        queue.add(request);
    }

    public interface BottomSheetListener {
        void onButtonClicked();
    }


}

