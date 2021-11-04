package com.example.refresh_selection;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonArrayRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.floatingactionbutton.FloatingActionButton;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

public class FragmentMain extends Fragment {
    SpaceCardAdapter spaceCardAdapter;
    private RecyclerView spaceRV;
    private ArrayList<SpaceCard> spaceCardArrayList=new ArrayList<SpaceCard>();//카드 정보 저장
//    private ViewGroup mainview;
    String area;
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        ViewGroup mainview =(ViewGroup)inflater.inflate(R.layout.activity_main3, container, false);
        requestSpace(mainview);//서버에 장소 요청



        FloatingActionButton fab = (FloatingActionButton) mainview.findViewById(R.id.fab_main);
        fab.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View view)
            {
                Intent intent =new Intent(mainview.getContext(),Calendar_java.class);
                startActivity(intent);
            }
        });
        ImageButton daejon_bt =(ImageButton)mainview.findViewById(R.id.deajon);
        ImageButton daegu_bt =(ImageButton)mainview.findViewById(R.id.deagu);
        ImageButton junju_bt =(ImageButton)mainview.findViewById(R.id.junju);
        ImageButton jeju_bt =(ImageButton)mainview.findViewById(R.id.jeju);

        daejon_bt.setOnClickListener(
                new View.OnClickListener(){

                    @Override
                    public void onClick(View v) {
                        area="대전";
                        Log.d("area",area);                    }
                }
        );
        daegu_bt.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                area="대구";
                Log.d("area",area);
            }
        });
        junju_bt.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                area="전주";
                Log.d("area",area);

            }
        });
        jeju_bt.setOnClickListener(new View.OnClickListener(){

            @Override
            public void onClick(View v) {
                area="제주";
                Log.d("area",area);

            }
        });
        return mainview;
    }


    public void makeSpaceCard(ViewGroup mainview) {
        spaceRV = (RecyclerView) mainview.findViewById(R.id.idRVSpace);//main_activity_3
        spaceRV.setHasFixedSize(true);
//        spaceCardArrayList.add(new SpaceCard("대청호1","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
//        spaceCardArrayList.add(new SpaceCard("대청호2","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));
//        spaceCardArrayList.add(new SpaceCard("대청호3","호수가 아름답다.","여기는 큰 호수이다.",R.drawable.test_img));

        spaceCardAdapter =new SpaceCardAdapter(mainview.getContext(),spaceCardArrayList);
        LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
        spaceRV.setLayoutManager(linearLayoutManager);
        spaceRV.setAdapter(spaceCardAdapter);
    }

    private void requestSpace(ViewGroup mainview) {//서버에 장소 좌표 요청하는 메소드
        RequestQueue queue = Volley.newRequestQueue(getContext());
        String url = "http://18.218.2.153:8089//api/getMainPlaces";//서버url
        JSONArray testjson = new JSONArray();//서버에게 전달 할 값을 json으로 보냄
//        try {
//            testjson.put(0,area);
//        } catch (JSONException e) {
//            e.printStackTrace();
//        }
        JsonArrayRequest request = new JsonArrayRequest(Request.Method.GET,url,testjson, new Response.Listener<JSONArray>() {
            @Override
            public void onResponse(JSONArray response) {
                Log.d("JSONArray",response.toString());
                try {
                    for(int i=0; i<response.length();i++){
                        JSONObject place =(JSONObject)response.get(i);
                        String name = place.getString("name");
                        String descript=place.getString("description");
                        String tag = place.getString("tag");
                        String ImgSrc = place.getString("imgSrc");

                        spaceCardArrayList.add(new SpaceCard(name,descript,tag,LoadImageFromWebOperations(ImgSrc,i+"")));
                    }
                    spaceRV = (RecyclerView) mainview.findViewById(R.id.idRVSpace);//main_activity_3
                    spaceRV.setHasFixedSize(true);

                    spaceCardAdapter =new SpaceCardAdapter(mainview.getContext(),spaceCardArrayList);
                    LinearLayoutManager linearLayoutManager = new LinearLayoutManager(mainview.getContext(), LinearLayoutManager.VERTICAL, false);
                    spaceRV.setLayoutManager(linearLayoutManager);
                    spaceRV.setAdapter(spaceCardAdapter);
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.d("responseError", "장소 가져오기 실패");
            }
        });
        request.setRetryPolicy(new DefaultRetryPolicy(DefaultRetryPolicy.DEFAULT_TIMEOUT_MS, DefaultRetryPolicy.DEFAULT_MAX_RETRIES, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        queue.add(request);
    }
    public static Drawable LoadImageFromWebOperations(String url,String srcName) {
        try {
            InputStream is = (InputStream) new URL(url).getContent();
            Drawable d = Drawable.createFromStream(is, "res/drawable/"+srcName);
            return d;
        } catch (Exception e) {
            return null;
        }
    }
}
