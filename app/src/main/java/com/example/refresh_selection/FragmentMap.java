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

public class FragmentMap extends Fragment implements MapView.MapViewEventListener, MapView.POIItemEventListener, ModalBottomSheet.BottomSheetListener {
    private int Tag;//마커 태그?
    private int Radius = 100;//반경?
    private GpsTracker gpsTracker;//gpsTracker 객체
    private float lati,longi;//위도,경도
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
        ViewGroup mapViewContainer = (ViewGroup) v.findViewById(R.id.map_view);//xml에 지도넣기
        mapViewContainer.addView(mapView);

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
        
        requestLatitudeLongtitude(v,mapView,lati,longi,Radius);//경도 위도 array를 받나?

//        testmarker = setMarker("충남대학교",36.369032616640105, 127.34697568537104,0);//setMarker(MarkerName, Latitude, Longitude, Tag);//서버에서 가져온 위도,경도,마커이름 //#태그번호는 임의로 지정?
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
        String url = String.format("http://3.19.218.150:8089//api/getPlaces?meters=%1$s&x=%2$s&y=%3$s",radius,longitude,latitude);//서버url

        JsonArrayRequest request = new JsonArrayRequest(url ,new Response.Listener<JSONArray>() {
            @Override
            public void onResponse(JSONArray response) {
                Log.d("JSONArray",response.toString());
//                testmarker=new MapPOIItem[response.length()];
                try {
//                    double [] l1={36.4001,36.4002,36.4004 };
//                    double[]l2={ 127.4005,127.39877,127.3969};
                    for (int i=0; i<response.length();i++) {
                        MapPOIItem marker = new MapPOIItem();
                        JSONObject place =(JSONObject)response.get(i);
                        String name = place.getString("poi_nm");
                        double Latitude= Double.parseDouble(place.getString("y"));//위도
                        double Longtitude = Double.parseDouble(place.getString("x"));//경도

                        marker.setMapPoint(MapPoint.mapPointWithGeoCoord(Latitude,Longtitude));
                        Log.d("길이",response.length()+"");
                        marker.setItemName(name);
                        marker.setMarkerType(MapPOIItem.MarkerType.BluePin); // 기본으로 제공하는 BluePin 마커 모양.
                        marker.setSelectedMarkerType(MapPOIItem.MarkerType.RedPin); // 마커를 클릭했을때, 기본으로 제공하는 RedPin 마커 모양.
                        markerArr.add(marker);
                        mapView.addPOIItem(markerArr.get(i));
                    }

                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }

        }, new Response.ErrorListener() {

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