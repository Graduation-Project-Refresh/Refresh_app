package com.example.refresh_selection;

import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonArrayRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.bottomsheet.BottomSheetBehavior;

import net.daum.mf.map.api.MapPOIItem;
import net.daum.mf.map.api.MapPoint;
import net.daum.mf.map.api.MapView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.RecyclerView;

public class FragmentMap extends Fragment implements MapView.MapViewEventListener, MapView.POIItemEventListener, ModalBottomSheet.BottomSheetListener {
    private String MarkerName;//마커 장소이름
    private double Latitude;//위도
    private double Longitude;//경도
    private int Tag;//마커 태그?
    private MapPOIItem[] testmarker;//마커객체?
    private int Radius = 50;//반경?
    private GpsTracker gpsTracker;//gpsTracker 객체
    private float lati,longi;


    //바텀시트
    private BottomSheetBehavior sheetBehavior;
    LinearLayout layoutBottomSheet;
    //바텀 시트 안에 카드뷰
    BottomCardAdapter bottomCardAdapter;
    private RecyclerView bottomRV;
    private ArrayList<BottomCard> bottomCardArrayList=new ArrayList<BottomCard>();//카드 정보 저장
    MapView mapView;
    ArrayList<MapPOIItem> markerArr = new ArrayList<MapPOIItem>();
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        View v = inflater.inflate(R.layout.activity_map, container, false);
        ViewGroup v2= (ViewGroup)inflater.inflate(R.layout.bottom_sheet, container, false);
        mapView = new MapView(getActivity());
        mapView.setPOIItemEventListener(this);//마커 눌렀을때만? 바텀시트 띄우기
        //showBottomSheetDialog();

        /**
         * bottom sheet state change listener
         * we are changing button text when sheet changed state
         * */
//        layoutBottomSheet=v2.findViewById(R.id.standard_bottom_sheet);
//
//        sheetBehavior = BottomSheetBehavior.from(layoutBottomSheet);
//
//        sheetBehavior.setBottomSheetCallback(new BottomSheetBehavior.BottomSheetCallback() {
//            @Override
//            public void onStateChanged(@NonNull View bottomSheet, int newState) {
//                switch (newState) {
//                    case BottomSheetBehavior.STATE_HIDDEN:
//                        break;
//                    case BottomSheetBehavior.STATE_EXPANDED: {
//
//                    }
//                    break;
//                    case BottomSheetBehavior.STATE_COLLAPSED: {
//
//                    }
//                    break;
//                    case BottomSheetBehavior.STATE_DRAGGING:
//                        break;
//                    case BottomSheetBehavior.STATE_SETTLING:
//                        break;
//                }
//            }
//
//            @Override
//            public void onSlide(@NonNull View bottomSheet, float slideOffset) {
//
//            }
//        });
        //바텀시트 행위


        //지도를 현재위치에 있는 장소로 띄어줌
        mapView.setMapViewEventListener(this);
        mapView.setCurrentLocationTrackingMode(MapView.CurrentLocationTrackingMode.TrackingModeOnWithHeading);
        //서버에 위도 경도요청하는 request보내기 반경을 요청하는 request 보내기
        //client가 반경을 줘야하나?
        mapView.getCurrentLocationTrackingMode();

        gpsTracker = new GpsTracker(getActivity());
         lati = (float)gpsTracker.getLatitude(); // 현재 위치 위도
         longi = (float)gpsTracker.getLongitude();// 현재위치 경도
        // 필요시 String address = getCurrentAddress(latitude, longitude);
        Log.d("lati longi",lati+" "+longi+"");
        
//        requestLatitudeLongtitude(v,mapView,lati,longi,Radius);//경도 위도 array를 받나?
        String []nm={"스타벅스","문지교회","CU"};
        double [] l1={36.400155695351465,36.400280953454406,36.40043247706835 };
        double[]l2={ 127.40052878490437,127.39877999700833,127.3969026539438};
        for (int i=0; i<3;i++) {
            MapPOIItem marker = new MapPOIItem();
            String name = nm[i];
            double Latitude=l1[i];
            double Longtitude = l2[i];
            marker.setMapPoint(MapPoint.mapPointWithGeoCoord(Latitude, Longitude));
            marker.setItemName(name);
            markerArr.add(marker);
        }
        mapView.addPOIItems(markerArr.toArray(new MapPOIItem[markerArr.size()]));
//        testmarker = setMarker("충남대학교",36.369032616640105, 127.34697568537104,0);//setMarker(MarkerName, Latitude, Longitude, Tag);//서버에서 가져온 위도,경도,마커이름 //#태그번호는 임의로 지정?
//        for(int i=0; i<testmarker.length;i++){
//            mapView.addPOIItem(testmarker[i]);//마커 표시
//        }
        ViewGroup mapViewContainer = (ViewGroup) v.findViewById(R.id.map_view);//xml에 지도넣기
            mapViewContainer.addView(mapView);



        return v;
    }

    private void showBottomSheetDialog() {//하단 모달 시트 띄우는 메소드
        ModalBottomSheet bottomSheet=new ModalBottomSheet();
        bottomSheet.show(getChildFragmentManager(),"bottomSheet");
    }

    @Override
    public void onButtonClicked() {//ModalBottomSheet.BottomSheetListner

    }


    private void requestLatitudeLongtitude(View v,MapView mapView,float latitude, float longitude,int radius) {//서버에 장소 좌표 요청하는 메소드
        RequestQueue queue = Volley.newRequestQueue(getContext());
        String url = String.format("http://18.218.2.153:8089//api/getPlaces?meters=%1$s&x=%2$s&y=%3$s",radius,longitude,latitude);//서버url
        JSONArray testjson = new JSONArray();//서버에게 전달 할 값을 json으로 보냄
        JSONObject test = new JSONObject();
        try {
            test.put("meters",radius);//
            test.put("x",longitude);//
            test.put("y",latitude);//

            //서버에게 위도 경도를 요청한다.
        } catch (JSONException e) {
            e.printStackTrace();
        }
        JsonArrayRequest request = new JsonArrayRequest(url ,new Response.Listener<JSONArray>() {
            @Override
            public void onResponse(JSONArray response) {
                Log.d("JSONArray",response.toString());
                testmarker=new MapPOIItem[response.length()];
                try {
//                    for(int i=0; i<response.length();i++){
//                        JSONObject place =(JSONObject)response.get(i);
//                        String name = place.getString("poi_nm");
//                        String descript=place.getString("mlsfc");
//                        String dis=place.getString("poi_nm");
//                        String price=place.getString("sgg_nm");
//                        double Latitude= place.getDouble("x");
//                        double Longtitude = place.getDouble("y");
//                        testmarker[i]=setMarker(name,latitude,longitude,i);
//                        //지도에 마커 표시
//
//                        String tag = place.getString("id");
////                        int ImgSrc = Integer.parseInt(place.getString("imgSrc"));
//                    }
//                    mapView.addPOIItems(testmarker);

                    for (int i=0; i<response.length();i++) {
                        MapPOIItem marker = new MapPOIItem();
                        JSONObject place =(JSONObject)response.get(i);
                        String name = place.getString("poi_nm");
                        double Latitude= place.getDouble("x");
                        double Longtitude = place.getDouble("y");
                        marker.setMapPoint(MapPoint.mapPointWithGeoCoord(latitude, longitude));
                        marker.setItemName(name);
                        markerArr.add(marker);
                    }

//                    MarkerName = jsonResponse.getString("markerName");//마커 이름
//                    Latitude = jsonResponse.getDouble("latitude");//마커 위도
//                    Longitude = jsonResponse.getDouble("longitude");//마커 경도
//                    Tag = jsonResponse.getInt("tag");//마커 구분 번호?
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }
//
//            @Override
//            public void onResponse(JSONObject response) {
//                Log.d("JSONArray",response.toString());
//                try {
//                    testmarker=new MapPOIItem[response.length()];
//                    for(int i=0; i<response.length();i++){
//                        JSONObject place =(JSONObject)response.get(i);
//                        String name = place.getString("bemd_nm");
//                        String descript=place.getString("lclas");
//                        String dis=place.getString("poi_nm");
//                        String price=place.getString("sgg_nm");
//                        double Latitude= place.getDouble("x");
//                        double Longtitude = place.getDouble("y");
//                        mapView.addPOIItem(setMarker(name,latitude,longitude,i));//지도에 마커 표시
//                        String tag = place.getString("id");
//
////                        int ImgSrc = Integer.parseInt(place.getString("imgSrc"));
//                        bottomCardArrayList.add(new BottomCard(name,descript,dis,price,R.drawable.test_img));//바텀 시트에 카드뷰생성
//                    }
////                    MarkerName = jsonResponse.getString("markerName");//마커 이름
////                    Latitude = jsonResponse.getDouble("latitude");//마커 위도
////                    Longitude = jsonResponse.getDouble("longitude");//마커 경도
////                    Tag = jsonResponse.getInt("tag");//마커 구분 번호?
//                } catch (JSONException e) {
//                    e.printStackTrace();
//                }
//            }
        }, new Response.ErrorListener() {
//            protected Map<String, Object> getParams() throws com.android.volley.AuthFailureError {
//                Map<String, Object> params = new HashMap<String, Object>();
//                params.put("meters",radius);//
//                params.put("x",longitude);//
//                params.put("y",latitude);//
//                return params;
//            };
            @Override
            public void onErrorResponse(VolleyError error) {
                Log.d("responseError", "위도, 경도 가져오기 실패 "+error.getMessage());
            }
        });
        request.setRetryPolicy(new DefaultRetryPolicy(DefaultRetryPolicy.DEFAULT_TIMEOUT_MS, DefaultRetryPolicy.DEFAULT_MAX_RETRIES, DefaultRetryPolicy.DEFAULT_BACKOFF_MULT));
        queue.add(request);
    }

    private MapPOIItem setMarker(String markerName, double latitude, double longitude, int tag) {
        //마커객체 생성 메소드
        MapPOIItem marker = new MapPOIItem();

        //맵 포인트 위도경도 설정 나중에 서버에서 제공하는 위도 ,경도로 설정
        MapPoint mapPoint = MapPoint.mapPointWithGeoCoord(latitude, longitude);//36.369015338611675, 127.34685766817087 충남대학교 좌표
        marker.setItemName(markerName);//마커이름
        marker.setTag(tag);//마커 태그?
        marker.setMapPoint(mapPoint);//마커를 위치에 표시
        marker.setMarkerType(MapPOIItem.MarkerType.BluePin); // 기본으로 제공하는 BluePin 마커 모양.
        marker.setSelectedMarkerType(MapPOIItem.MarkerType.RedPin); // 마커를 클릭했을때, 기본으로 제공하는 RedPin 마커 모양.
        return marker;
    }

    //MapView.MapViewEventListener 구현
    @Override
    public void onMapViewInitialized(MapView mapView) {

    }

    @Override
    public void onMapViewCenterPointMoved(MapView mapView, MapPoint mapCenterPoint) {

    }

    @Override
    public void onMapViewZoomLevelChanged(MapView mapView, int i) {

    }

    @Override
    public void onMapViewSingleTapped(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDoubleTapped(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewLongPressed(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDragStarted(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewDragEnded(MapView mapView, MapPoint mapPoint) {

    }

    @Override
    public void onMapViewMoveFinished(MapView mapView, MapPoint mapPoint) {

    }

//MapView.POIItemEventListener 구현
    @Override
    public void onPOIItemSelected(MapView mapView, MapPOIItem mapPOIItem) {
        showBottomSheetDialog();
        Log.d("marker click","마커 눌럿음");
    }

    @Override
    public void onCalloutBalloonOfPOIItemTouched(MapView mapView, MapPOIItem mapPOIItem) {

    }

    @Override
    public void onCalloutBalloonOfPOIItemTouched(MapView mapView, MapPOIItem mapPOIItem, MapPOIItem.CalloutBalloonButtonType calloutBalloonButtonType) {

    }

    @Override
    public void onDraggablePOIItemMoved(MapView mapView, MapPOIItem mapPOIItem, MapPoint mapPoint) {

    }

    public float getLati() {
        return lati;
    }

    public void setLati(float lati) {
        this.lati = lati;
    }

    public float getLongi() {
        return longi;
    }

    public void setLongi(float longi) {
        this.longi = longi;
    }
}