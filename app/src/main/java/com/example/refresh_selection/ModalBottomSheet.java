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
    private ArrayList<BottomCard> bottomCardArrayList= new ArrayList<>();//카드 정보 저장
    private int Radius =300;

    public ModalBottomSheet(){//empty constructor

    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }

    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
//        mListener = (BottomSheetListener) getContext();
        view = inflater.inflate(R.layout.bottom_sheet, container, false);
        FragmentMap fm = new FragmentMap();
        requestLatitudeLongtitude(fm.getLati(),fm.getLongi(),Radius);
        makeMapSpaceCard(view);
        return view;
    }

    //장소 카드 만듣
    public void makeMapSpaceCard(View mainview) {
        bottomRV = (RecyclerView) mainview.findViewById(R.id.bottom_sheet_recycleview);//activitiy_map : bottomsheet layout
        bottomRV.setHasFixedSize(true);

//        bottomCardArrayList.add(new BottomCard("대청호1","호수가 아름답다.","300m","1000원",R.drawable.test_img));
//        bottomCardArrayList.add(new BottomCard("대청호1","호수가 아름답다.","300m","1000원",R.drawable.test_img));
//        bottomCardArrayList.add(new BottomCard("대청호1","호수가 아름답다.","300m","1000원",R.drawable.test_img));
//        Context con = mainview.getChildAt(1).getContext();
//        Log.d("context",con+"");
        bottomCardAdapter =new BottomCardAdapter(mainview.getContext(),bottomCardArrayList);//content가 문제인거 같은데..
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        bottomRV.setLayoutManager(linearLayoutManager);
        bottomRV.setAdapter(bottomCardAdapter);
    }
    private void requestLatitudeLongtitude( float latitude, float longitude, int radius) {//서버에 장소 좌표 요청하는 메소드
        RequestQueue queue = Volley.newRequestQueue(getContext());
        String url = String.format("http://18.218.2.153:8089//api/getPlaces?meters=%1$s&x=%2$s&y=%3$s",radius,longitude,latitude);//서버url
        JsonArrayRequest request = new JsonArrayRequest(url ,new Response.Listener<JSONArray>() {
            @Override
            public void onResponse(JSONArray response) {
                Log.d("bottom_JSONArray",response.toString());
                try {
                    for(int i=0; i<response.length();i++){
                        JSONObject place =(JSONObject)response.get(i);
                        String name = place.getString("bemd_nm");
                        String descript=place.getString("lclas");
                        String dis=place.getString("poi_nm");
                        String price=place.getString("sgg_nm");
                        String tag = place.getString("id");
                        BottomCard cd=new BottomCard(name,descript,dis,price,R.drawable.test_img);
//                        int ImgSrc = Integer.parseInt(place.getString("imgSrc"));
                        bottomCardArrayList.add(cd);//바텀 시트에 카드뷰생성
                    }
//                    Tag = jsonResponse.getInt("tag");//마커 구분 번호?
                } catch (JSONException e) {
                    e.printStackTrace();
                }
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
